"""Data preflight regressions; no Hub access or model downloads required.

Run with the recipe dependencies and pytest installed:
    python -m pytest tests/test_train_setfit.py
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from datasets import ClassLabel, Dataset, Features, Value

SCRIPT = Path(__file__).resolve().parents[1] / "classification" / "train-setfit.py"
spec = importlib.util.spec_from_file_location("train_setfit", SCRIPT)
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)


def typed_dataset(labels):
    return Dataset.from_dict(
        {"text": [f"document {i}" for i in range(len(labels))], "label": labels},
        features=Features({"text": Value("string"), "label": ClassLabel(names=["a", "b", "c"])}),
    )


def test_missing_classlabel_is_dropped_without_remapping():
    cleaned = recipe.prepare_split(typed_dataset([0, -1, 2, None]), "text", "label", "train")
    assert list(cleaned["label"]) == [0, 2]
    assert recipe.resolve_label_names(cleaned, "label") == ["a", "b", "c"]


def test_plain_negative_integer_is_a_real_class():
    data = Dataset.from_dict({"text": ["a", "b", "c"], "label": [-1, 0, 1]})
    assert list(recipe.prepare_split(data, "text", "label", "train")["label"]) == ["-1", "0", "1"]


def test_nan_is_removed_before_string_cast():
    data = Dataset.from_dict({"text": ["a", "b", "c"], "label": [1.0, float("nan"), 2.0]})
    cleaned = recipe.prepare_split(data, "text", "label", "train")
    assert len(cleaned) == 2
    assert "nan" not in cleaned["label"]


@pytest.mark.parametrize("labels", [[], [None, None], ["", " "]])
def test_empty_or_unlabelled_split_is_rejected(labels):
    data = Dataset.from_dict({"text": ["document"] * len(labels), "label": labels})
    with pytest.raises(SystemExit, match="empty|No labelled rows"):
        recipe.prepare_split(data, "text", "label", "eval")


@pytest.mark.parametrize("text", [None, "", " "])
def test_empty_text_split_is_rejected_before_training(text):
    data = Dataset.from_dict({"text": [text], "label": ["a"]})
    with pytest.raises(SystemExit, match="No usable text rows"):
        recipe.prepare_split(data, "text", "label", "eval")


@pytest.mark.parametrize("text", [None, "", " "])
def test_missing_text_is_dropped_without_rejecting_usable_rows(text):
    data = Dataset.from_dict({"text": [text, "a real document"], "label": ["a", "b"]})
    cleaned = recipe.prepare_split(data, "text", "label", "eval")
    assert list(cleaned["text"]) == ["a real document"]
    assert list(cleaned["label"]) == ["b"]


def test_non_string_text_is_rejected_before_training():
    data = Dataset.from_dict({"text": [12], "label": ["a"]})
    with pytest.raises(SystemExit, match="Clean the text column"):
        recipe.prepare_split(data, "text", "label", "eval")


@pytest.mark.parametrize("column", ["text", "label"])
def test_missing_columns_are_actionable(column):
    data = Dataset.from_dict({"text": ["document"], "label": ["a"]}).remove_columns(column)
    with pytest.raises(SystemExit, match=f"--{column}-column"):
        recipe.prepare_split(data, "text", "label", "eval")


def test_multilabel_eval_is_rejected():
    data = Dataset.from_dict({"text": ["document"], "label": [["a", "b"]]})
    with pytest.raises(SystemExit, match="multi-label"):
        recipe.prepare_split(data, "text", "label", "eval")


@pytest.mark.parametrize("labels", [["a", "a", "b", "b"], [-1, -1, 2, 2]])
def test_plain_labels_get_stratified_disjoint_carve(monkeypatch, labels):
    data = Dataset.from_dict({"text": ["one", "two", "three", "four"], "label": labels})
    monkeypatch.setattr(recipe, "load_dataset", lambda *args, **kwargs: data)
    train, evaluation = recipe.split_train_eval("fixture", None, "train", None, 0.5, 2, "label")
    assert len(set(train["label"])) == len(set(evaluation["label"])) == 2
    assert set(train["text"]).isdisjoint(evaluation["text"])
    assert set(train.features["label"].names) == {str(label) for label in labels}


def test_carve_drops_missing_classlabel_before_stratifying(monkeypatch):
    data = typed_dataset([0, 0, 2, 2, -1, None])
    monkeypatch.setattr(recipe, "load_dataset", lambda *args, **kwargs: data)
    train, evaluation = recipe.split_train_eval("fixture", None, "train", None, 0.5, 2, "label")
    assert len(train) == len(evaluation) == 2
    assert set(train["label"]) == set(evaluation["label"]) == {0, 2}


def test_one_observed_class_stops_before_model_loading(monkeypatch):
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "fixture", "user/model", "--hf-token", "fake"])
    monkeypatch.setattr(recipe, "login", Mock())
    monkeypatch.setattr(recipe, "HfApi", Mock())
    monkeypatch.setattr(recipe, "pick_eval_split", lambda *args: "test")
    monkeypatch.setattr(recipe, "split_train_eval", lambda *args: (typed_dataset([2, 2]), typed_dataset([0, 2])))
    load_model = Mock()
    monkeypatch.setattr(recipe.SetFitModel, "from_pretrained", load_model)
    with pytest.raises(SystemExit, match="two observed classes"):
        recipe.main(recipe.parse_args())
    load_model.assert_not_called()


def test_evaluation_decodes_its_own_classlabel_table():
    evaluation = typed_dataset([2, 0])
    model = Mock()
    model.predict.return_value = ["c", "a"]
    assert recipe.evaluate(model, evaluation, "text", "label")["accuracy"] == 1.0


def test_help_renders_slice_example(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "--help"])
    with pytest.raises(SystemExit) as result:
        recipe.parse_args()
    assert result.value.code == 0
    assert "train[:10%]" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("job_id", "accelerator", "cuda", "expected"),
    [
        ("job-123", "l40sx1", True, "l40sx1"),
        ("job-123", "a100-large", True, "a100-large"),
        ("job-123", "none", False, "cpu-basic"),
        ("job-123", "", False, "cpu-basic"),
        ("", "l40sx1", True, "t4-small"),
    ],
)
def test_reproduction_preserves_jobs_gpu_flavor(monkeypatch, job_id, accelerator, cuda, expected):
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "fixture", "user/model"])
    monkeypatch.setenv("JOB_ID", job_id)
    monkeypatch.setenv("ACCELERATOR", accelerator)
    monkeypatch.setattr(recipe.torch.cuda, "is_available", lambda: cuda)
    command = recipe.build_reproduce_command(recipe.parse_args())
    assert command.startswith(f"hf jobs uv run --flavor {expected} --secrets HF_TOKEN")


def _flavor(name, cpu, ram, model=None, quantity=None):
    accelerator = None if model is None else SimpleNamespace(model=model, quantity=str(quantity))
    return SimpleNamespace(name=name, cpu=f"{cpu} vCPU", ram=f"{ram} GB", accelerator=accelerator)


# A subset of HfApi().list_jobs_hardware(), as listed on 2026-10-02.
JOBS_HARDWARE = [
    _flavor("cpu-basic", 2, 16),
    _flavor("cpu-upgrade", 8, 32),
    _flavor("t4-small", 4, 15, "T4", 1),
    _flavor("t4-medium", 8, 30, "T4", 1),
    _flavor("a10g-small", 4, 15, "A10G", 1),
    _flavor("a10g-large", 12, 46, "A10G", 1),
    _flavor("a10g-largex2", 24, 92, "A10G", 2),
    _flavor("l4x1", 8, 30, "L4", 1),
    _flavor("l4x4", 48, 186, "L4", 4),
    _flavor("l40sx1", 8, 62, "L40S", 1),
    _flavor("h200x8", 184, 2048, "H200", 8),
]


@pytest.fixture(autouse=True)
def _offline_jobs_hardware(monkeypatch):
    """jobs_flavor() reads the public Jobs hardware list; tests use a recorded subset instead."""
    monkeypatch.setattr("huggingface_hub.HfApi.list_jobs_hardware", lambda self, token=None: JOBS_HARDWARE)
    monkeypatch.delenv("CPU_CORES", raising=False)
    monkeypatch.delenv("MEMORY", raising=False)


@pytest.mark.parametrize(
    ("gpu_name", "gpu_count", "cpu_cores", "memory", "expected"),
    [
        ("Tesla T4", 1, "4", "15.0G", "t4-small"),
        ("NVIDIA A10G", 1, "3", "15.0G", "a10g-small"),  # Jobs reports 3 cores on a10g-small
        ("NVIDIA A10G", 1, "12", "46.0G", "a10g-large"),
        ("NVIDIA A10G", 2, "24", "99.0G", "a10g-largex2"),  # Jobs reports 99G, the listing says 92
        ("NVIDIA L40S", 1, "8", "62.0G", "l40sx1"),
        ("NVIDIA L4", 4, "48", "185.0G", "l4x4"),
        ("NVIDIA H200", 8, "184", "2048.0G", "h200x8"),
    ],
)
def test_reproduction_names_flavor_when_accelerator_is_bare_gpu(
    monkeypatch, gpu_name, gpu_count, cpu_cores, memory, expected
):
    """On Jobs, ACCELERATOR is "gpu", not the flavor: the flavor comes from the hardware list."""
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "fixture", "user/model"])
    monkeypatch.setenv("JOB_ID", "job-123")
    monkeypatch.setenv("ACCELERATOR", "gpu")
    monkeypatch.setenv("CPU_CORES", cpu_cores)
    monkeypatch.setenv("MEMORY", memory)
    monkeypatch.setattr(recipe.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(recipe.torch.cuda, "get_device_name", lambda index=0: gpu_name)
    monkeypatch.setattr(recipe.torch.cuda, "device_count", lambda: gpu_count)
    command = recipe.build_reproduce_command(recipe.parse_args())
    assert command.startswith(f"hf jobs uv run --flavor {expected} --secrets HF_TOKEN")


def test_reproduction_names_cpu_flavor(monkeypatch):
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "fixture", "user/model"])
    monkeypatch.setenv("JOB_ID", "job-123")
    monkeypatch.setenv("ACCELERATOR", "cpu")
    monkeypatch.setenv("CPU_CORES", "8")
    monkeypatch.setenv("MEMORY", "32.0G")
    monkeypatch.setattr(recipe.torch.cuda, "is_available", lambda: False)
    command = recipe.build_reproduce_command(recipe.parse_args())
    assert command.startswith("hf jobs uv run --flavor cpu-upgrade --secrets HF_TOKEN")


def test_reproduction_falls_back_when_hardware_lookup_fails(monkeypatch):
    def offline(self, token=None):
        raise OSError("no network")

    monkeypatch.setattr("huggingface_hub.HfApi.list_jobs_hardware", offline)
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "fixture", "user/model"])
    monkeypatch.setenv("JOB_ID", "job-123")
    monkeypatch.setenv("ACCELERATOR", "gpu")
    monkeypatch.setattr(recipe.torch.cuda, "is_available", lambda: True)
    command = recipe.build_reproduce_command(recipe.parse_args())
    assert command.startswith("hf jobs uv run --flavor t4-small --secrets HF_TOKEN")


@pytest.mark.parametrize(("nvidia_smi", "expected"), [("NVIDIA L4\n", "l4x1"), (None, "")])
def test_jobs_flavor_without_torch(monkeypatch, nvidia_smi, expected):
    """Torch-free drivers (the -server recipes) read the GPU from nvidia-smi; a GPU run
    whose GPU cannot be identified gets "", never a CPU flavor."""
    import subprocess
    import sys

    def fake_run(*args, **kwargs):
        if nvidia_smi is None:
            raise FileNotFoundError("nvidia-smi")
        return SimpleNamespace(stdout=nvidia_smi)

    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("ACCELERATOR", "gpu")
    monkeypatch.setenv("CPU_CORES", "8")
    monkeypatch.setenv("MEMORY", "30.0G")
    assert recipe.jobs_flavor() == expected


@pytest.mark.parametrize("is_private", [False, True])
def test_private_destination_visibility_checked_before_loading_data(monkeypatch, is_private):
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "fixture", "user/model", "--private", "--hf-token", "fake"])
    monkeypatch.setattr(recipe, "login", Mock())
    api = Mock()
    api.model_info.return_value.private = is_private
    monkeypatch.setattr(recipe, "HfApi", Mock(return_value=api))
    load_data = Mock(side_effect=RuntimeError("data loading reached"))
    monkeypatch.setattr(recipe, "pick_eval_split", load_data)
    error = RuntimeError if is_private else SystemExit
    message = "data loading reached" if is_private else "is public"
    with pytest.raises(error, match=message):
        recipe.main(recipe.parse_args())
    assert load_data.call_count == int(is_private)
    api.create_repo.assert_called_once_with("user/model", repo_type="model", private=True, exist_ok=True)
