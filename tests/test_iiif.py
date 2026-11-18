"""Local publish, manifest and check runs; no Hub access.

Run with the recipe dependencies and pytest installed:
    uv run --python 3.13 --with pytest --with "pyvips[binary]" --with pyarrow --with huggingface-hub \
        python -m pytest tests/test_iiif.py
"""

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

pyvips = pytest.importorskip("pyvips")
pq = pytest.importorskip("pyarrow.parquet")

SCRIPT = Path(__file__).resolve().parents[1] / "iiif-tiles" / "iiif.py"
spec = importlib.util.spec_from_file_location("iiif_recipe", SCRIPT)
recipe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = recipe  # dataclasses look the module up while the script loads
spec.loader.exec_module(recipe)

BASE = "http://localhost:8000"
PUBLIC_DOMAIN = "http://creativecommons.org/publicdomain/mark/1.0/"


def picture(width, height, bands=3):
    """A gradient image, so resized copies are not a flat colour."""
    image = pyvips.Image.xyz(width, height).cast("uchar")
    if bands == 1:
        return image[0]
    return image[0].bandjoin([image[1], image[0]])


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    (root / "Book A").mkdir(parents=True)
    (root / "book-b").mkdir()
    picture(3000, 2200).jpegsave(str(root / "Book A" / "page2.jpg"))
    picture(3000, 2200).jpegsave(str(root / "Book A" / "page10.jpg"))
    picture(900, 1200).jpegsave(str(root / "book-b" / "page2.jpg"))  # same stem as in Book A
    picture(300, 200).bandjoin(255).pngsave(str(root / "small.png"))  # has alpha
    sideways = picture(800, 600).copy()
    sideways.set_type(pyvips.GValue.gint_type, "orientation", 6)
    sideways.jpegsave(str(root / "rotated.jpg"))
    (root / "notes.txt").write_text("not an image")
    (root / "metadata.csv").write_text(
        "file_name,label,date,rights\n"
        "Book A/page2.jpg,Page two,1936,\n"
        f"rotated.jpg,Rotated photo,,{PUBLIC_DOMAIN}\n"
        "missing.jpg,No such file,,\n"
    )
    (root / "items.csv").write_text("folder,label,author\nBook A,The first book,A. Writer\n")
    return root


def run(*argv):
    return recipe.main([str(part) for part in argv])


def read_json(path):
    return json.loads(Path(path).read_text())


def test_slugify_is_ascii_and_keeps_different_names_apart():
    assert recipe.slugify("Book A (1936)") == "book-a-1936"
    assert recipe.slugify("Rätsel") == "ratsel"
    first, second = recipe.slugify("页1"), recipe.slugify("图1")
    assert first != second
    assert all(ch.isascii() and (ch.isalnum() or ch == "-") for ch in first + second)


def test_image_ids_join_folder_and_stem():
    assert recipe.image_id_for("Book A/page2.jpg") == "book-a__page2"
    assert recipe.image_id_for("page2.jpg") == "page2"
    assert recipe.item_id_for("") == "_root"


def test_fit_long_edge_rounds_the_short_edge_half_up():
    assert recipe.fit_long_edge(2411, 3372, 1024) == (732, 1024)
    assert recipe.fit_long_edge(3, 2, 2) == (2, 1)
    assert recipe.fit_long_edge(300, 200, 400) is None


def test_natural_order_puts_page2_before_page10():
    names = sorted(["page10", "page2", "page1"], key=recipe.natural_key)
    assert names == ["page1", "page2", "page10"]


def test_publish_writes_a_consistent_collection(source, tmp_path):
    dest = tmp_path / "published"
    assert run("publish", source, dest, "--base-url", BASE, "--title", "Test") == 0

    collection = read_json(dest / "collection.json")
    assert collection["id"] == f"{BASE}/collection.json"
    assert [item["id"].rsplit("/", 1)[1] for item in collection["items"]] == [
        "_root.json",
        "book-a.json",
        "book-b.json",
    ]

    book_a = read_json(dest / "manifests" / "book-a.json")
    assert book_a["label"] == {"none": ["The first book"]}
    assert book_a["metadata"][0]["value"] == {"none": ["A. Writer"]}
    assert [canvas["label"]["none"][0] for canvas in book_a["items"]] == ["Page two", "page10"]
    # The canvas id holds the image id only, so regrouping images does not change it.
    assert book_a["items"][0]["id"] == f"{BASE}/canvas/book-a__page2"

    assert run("check", dest) == 0


def test_rotated_image_is_stored_upright(source, tmp_path):
    dest = tmp_path / "published"
    run("publish", source, dest, "--base-url", BASE)
    info = read_json(dest / "images" / "rotated" / "info.json")
    assert (info["width"], info["height"]) == (600, 800)
    full = pyvips.Image.new_from_file(str(dest / "images" / "rotated" / "full" / "max" / "0" / "default.jpg"))
    assert (full.width, full.height) == (600, 800)
    assert full.get_typeof("orientation") == 0 or full.get("orientation") == 1


def test_small_image_has_one_whole_image_tile_and_large_image_has_tiles(source, tmp_path):
    dest = tmp_path / "published"
    run("publish", source, dest, "--base-url", BASE)

    small = read_json(dest / "images" / "small" / "info.json")
    assert small["tiles"] == [{"width": 300, "scaleFactors": [1]}]
    assert small["sizes"] == [{"width": 300, "height": 200}]
    flattened = pyvips.Image.new_from_file(str(dest / "images" / "small" / "full" / "max" / "0" / "default.jpg"))
    assert flattened.bands == 3

    large = read_json(dest / "images" / "book-a__page2" / "info.json")
    assert large["tiles"][0]["width"] == 512
    assert (dest / "images" / "book-a__page2" / "0,0,512,512" / "512,512" / "0" / "default.jpg").is_file()
    # Thumbnail and derivative exist as files but are not tile levels, so `sizes` omits them.
    assert (dest / "images" / "book-a__page2" / "full" / "400,293" / "0" / "default.jpg").is_file()
    assert (dest / "images" / "book-a__page2" / "full" / "1024,751" / "0" / "default.jpg").is_file()
    assert {"width": 1024, "height": 751} not in large["sizes"]


def test_index_has_one_row_per_image_with_metadata(source, tmp_path):
    dest = tmp_path / "published"
    run("publish", source, dest, "--base-url", BASE)
    rows = pq.read_table(io.BytesIO((dest / "index.parquet").read_bytes())).to_pylist()
    by_id = {row["image_id"]: row for row in rows}
    assert len(rows) == 5
    assert by_id["book-a__page2"]["meta_date"] == "1936"
    assert by_id["book-a__page2"]["derivative_url"] == f"{BASE}/images/book-a__page2/full/1024,751/0/default.jpg"
    assert by_id["small"]["derivative_url"] == f"{BASE}/images/small/full/max/0/default.jpg"
    assert by_id["rotated"]["rights"] == PUBLIC_DOMAIN


def test_second_publish_skips_unchanged_images(source, tmp_path, capsys):
    dest = tmp_path / "published"
    run("publish", source, dest, "--base-url", BASE)
    capsys.readouterr()
    run("publish", source, dest, "--base-url", BASE, "--json")
    result = json.loads(capsys.readouterr().out)
    assert (result["processed"], result["skipped"]) == (0, 5)


def test_manifest_rewrites_ids_for_a_new_base_url_without_the_source(source, tmp_path):
    dest = tmp_path / "published"
    run("publish", source, dest, "--base-url", BASE)
    tile = dest / "images" / "book-a__page2" / "0,0,512,512" / "512,512" / "0" / "default.jpg"
    tile_time = tile.stat().st_mtime_ns

    assert run("manifest", dest, "--base-url", "https://example.org/iiif", "--title", "Renamed") == 0
    assert read_json(dest / "images" / "small" / "info.json")["id"] == "https://example.org/iiif/images/small"
    assert read_json(dest / "collection.json")["label"] == {"none": ["Renamed"]}
    assert read_json(dest / "manifests" / "_root.json")["label"] == {"none": ["Renamed"]}
    assert tile.stat().st_mtime_ns == tile_time  # no image was written again
    assert run("check", dest) == 0


def test_check_reports_a_missing_file(source, tmp_path, capsys):
    dest = tmp_path / "published"
    run("publish", source, dest, "--base-url", BASE)
    (dest / "images" / "small" / "full" / "max" / "0" / "default.jpg").unlink()
    capsys.readouterr()
    assert run("check", dest, "--json") == 1
    assert any("small" in problem for problem in json.loads(capsys.readouterr().out)["problems"])


def test_two_files_with_the_same_id_stop_the_run(source, tmp_path, capsys):
    picture(100, 100).pngsave(str(source / "Book A" / "page2.png"))  # same id as page2.jpg
    assert run("publish", source, tmp_path / "published", "--base-url", BASE) == 2
    assert "same image id" in capsys.readouterr().err


def test_rights_must_be_a_known_uri(source, tmp_path, capsys):
    assert run("publish", source, tmp_path / "published", "--base-url", BASE, "--rights", "public domain") == 2
    assert "rights must be" in capsys.readouterr().err


def test_local_destination_needs_a_base_url(source, tmp_path, capsys):
    assert run("publish", source, tmp_path / "published") == 2
    assert "--base-url" in capsys.readouterr().err


def test_other_cloud_storage_is_refused_with_a_next_step(tmp_path, capsys):
    assert run("publish", "s3://bucket/scans", tmp_path / "published", "--base-url", BASE) == 2
    assert "Hugging Face bucket" in capsys.readouterr().err


def test_metadata_file_in_a_subfolder_uses_paths_relative_to_it(source, tmp_path):
    (source / "book-b" / "metadata.csv").write_text("file_name,label\npage2.jpg,Second book page\n")
    dest = tmp_path / "published"
    run("publish", source, dest, "--base-url", BASE)
    book_b = read_json(dest / "manifests" / "book-b.json")
    assert book_b["items"][0]["label"] == {"none": ["Second book page"]}


def test_two_folders_with_the_same_id_stop_the_run(source, tmp_path, capsys):
    (source / "BOOK-B").mkdir(exist_ok=True)
    (source / "Book B").mkdir()
    picture(100, 100).jpegsave(str(source / "Book B" / "other.jpg"))
    assert run("publish", source, tmp_path / "published", "--base-url", BASE) == 2
    assert "same item id" in capsys.readouterr().err


def test_default_rights_reach_the_index_and_can_be_cleared(source, tmp_path):
    dest = tmp_path / "published"
    run("publish", source, dest, "--base-url", BASE, "--rights", PUBLIC_DOMAIN)
    rows = pq.read_table(io.BytesIO((dest / "index.parquet").read_bytes())).to_pylist()
    assert all(row["rights"] == PUBLIC_DOMAIN for row in rows)
    assert read_json(dest / "collection.json")["rights"] == PUBLIC_DOMAIN

    run("manifest", dest, "--rights", "")
    assert "rights" not in read_json(dest / "collection.json")


def test_destination_inside_the_source_is_refused(source, capsys):
    assert run("publish", source, source / "out", "--base-url", BASE) == 2
    assert "overlap" in capsys.readouterr().err


def test_check_reports_a_wrong_index_url_and_a_missing_last_tile(source, tmp_path, capsys):
    dest = tmp_path / "published"
    run("publish", source, dest, "--base-url", BASE)
    (dest / "images" / "book-a__page2" / "2560,2048,440,152" / "440,152" / "0" / "default.jpg").unlink()
    table = pq.read_table(io.BytesIO((dest / "index.parquet").read_bytes()))
    rows = table.to_pylist()
    rows[0]["image_url"] = "http://localhost:8000/images/wrong/full/max/0/default.jpg"
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist(rows), str(dest / "index.parquet"))
    capsys.readouterr()
    assert run("check", dest, "--json") == 1
    problems = " ".join(json.loads(capsys.readouterr().out)["problems"])
    assert "last tile" in problems and "index image_url" in problems


def test_leftover_files_from_a_stopped_run_are_cleared(source, tmp_path):
    dest = tmp_path / "published"
    stale = dest / "images" / "small" / "0,0,256,256" / "256,256" / "0" / "default.jpg"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"left over from a run with other settings")
    run("publish", source, dest, "--base-url", BASE)
    assert not stale.exists()
    assert run("check", dest) == 0


def test_image_that_cannot_be_read_fails_alone(source, tmp_path, capsys):
    (source / "broken.jpg").write_bytes(b"this is not a JPEG")
    dest = tmp_path / "published"
    assert run("publish", source, dest, "--base-url", BASE, "--json") == 1
    result = json.loads(capsys.readouterr().out)
    assert [failure["source"] for failure in result["failed"]] == ["broken.jpg"]
    assert result["images"] == 5
    assert run("check", dest) == 0


def test_space_readme_title_is_valid_yaml_for_awkward_titles():
    title = 'A "quoted" title with a backslash \\ and\na new line'
    line = recipe.space_readme(title, "https://example.org/collection.json", "collection", "https://example.org")
    front_matter = line.split("---")[1]
    title_line = next(row for row in front_matter.splitlines() if row.startswith("title: "))
    assert json.loads(title_line[len("title: ") :]) == 'A "quoted" title with a backslash \\ and a new line'
