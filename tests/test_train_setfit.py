"""Data preflight regressions; no Hub access or model downloads required.

Run with the recipe dependencies and pytest installed:
    python -m pytest tests/test_train_setfit.py
"""

import importlib.util
from pathlib import Path
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


@pytest.fixture(autouse=True)
def _clear_jobs_flavor(monkeypatch):
    monkeypatch.delenv("HF_JOB_FLAVOR", raising=False)  # set when the tests themselves run inside a Job


@pytest.mark.parametrize(
    ("job_id", "flavor", "cuda", "expected"),
    [
        ("job-123", "l40sx1", True, "l40sx1"),
        ("job-123", "cpu-upgrade", False, "cpu-upgrade"),
        ("job-123", "", True, "t4-small"),  # HF_JOB_FLAVOR unset: the suggested default
        ("", "l40sx1", True, "t4-small"),  # local run: never claims a Jobs flavor
    ],
)
def test_reproduction_uses_hf_job_flavor(monkeypatch, job_id, flavor, cuda, expected):
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "fixture", "user/model"])
    monkeypatch.setenv("HF_JOB_ID", job_id)
    monkeypatch.setenv("HF_JOB_FLAVOR", flavor)
    monkeypatch.setattr(recipe.torch.cuda, "is_available", lambda: cuda)
    command = recipe.build_reproduce_command(recipe.parse_args())
    assert command.startswith(f"hf jobs uv run --flavor {expected} --secrets HF_TOKEN")


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
