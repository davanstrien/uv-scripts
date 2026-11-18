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
    assert recipe.fit_long_edge(2, 3, 2) == (1, 2)  # 1.33 rounds down
    assert recipe.fit_long_edge(5, 8, 4) == (3, 4)  # 2.5 rounds up
    assert recipe.fit_long_edge(8, 5, 4) == (4, 3)
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


def published_image(dest, image_id):
    return pyvips.Image.new_from_file(str(dest / "images" / image_id / "full" / "max" / "0" / "default.jpg"))


def test_transparent_pixels_become_white(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    black = pyvips.Image.black(200, 100, bands=3).cast("uchar")
    black.bandjoin(0).pngsave(str(source / "clear.png"))  # fully transparent black
    run("publish", source, tmp_path / "published", "--base-url", BASE)
    assert published_image(tmp_path / "published", "clear").avg() > 250


def test_embedded_colour_profile_is_applied(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    original = picture(400, 300)
    wide = original.icc_transform("p3", input_profile="srgb")  # same colours, Display P3 numbers
    assert wide.get_typeof("icc-profile-data") != 0
    wide.tiffsave(str(source / "wide.tif"))
    run("publish", source, tmp_path / "published", "--base-url", BASE)
    result = published_image(tmp_path / "published", "wide")
    untreated = (wide.cast("int") - original.cast("int")).abs().avg()
    treated = (result.cast("int") - original.cast("int")).abs().avg()
    assert treated < 3 and treated < untreated / 2


def test_multi_page_tiff_warns_but_pyramid_does_not(tmp_path, capsys):
    source = tmp_path / "source"
    source.mkdir()
    pages = pyvips.Image.arrayjoin([picture(600, 400), picture(600, 400)], across=1).copy()
    pages.set_type(pyvips.GValue.gint_type, "page-height", 400)
    pages.tiffsave(str(source / "document.tif"))
    picture(1200, 800).tiffsave(str(source / "pyramid.tif"), pyramid=True, tile=True, tile_width=256, tile_height=256)
    run("publish", source, tmp_path / "published", "--base-url", BASE)
    warnings = [line for line in capsys.readouterr().err.splitlines() if "multi-page" in line]
    assert len(warnings) == 1 and "document.tif" in warnings[0]


def test_removed_item_loses_its_manifest_file(source, tmp_path):
    dest = tmp_path / "published"
    run("publish", source, dest, "--base-url", BASE)
    for path in (source / "book-b").iterdir():
        path.unlink()
    (source / "book-b").rmdir()
    run("publish", source, dest, "--base-url", BASE)
    assert not (dest / "manifests" / "book-b.json").exists()
    run("manifest", dest, "--base-url", "https://example.org/new")
    leftovers = [p for p in dest.rglob("*.json") if "localhost:8000" in p.read_text() and "_publish" not in p.parts]
    assert leftovers == []


def test_when_every_replacement_fails_the_collection_is_emptied(source, tmp_path, monkeypatch, capsys):
    dest = tmp_path / "published"
    run("publish", source, dest, "--base-url", BASE)

    def no_space(self, local_dir, dest_prefix):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(recipe.LocalDest, "put_tree", no_space)
    capsys.readouterr()
    assert run("publish", source, dest, "--base-url", BASE, "--tile-size", "256", "--json") == 1
    assert len(json.loads(capsys.readouterr().out)["failed"]) == 5
    assert read_json(dest / "collection.json")["items"] == []
    assert pq.read_table(io.BytesIO((dest / "index.parquet").read_bytes())).num_rows == 0
    monkeypatch.undo()
    assert run("publish", source, dest, "--base-url", BASE) == 0  # a later run recovers
    assert run("check", dest) == 0


def test_published_image_that_becomes_unreadable_stays_out_until_fixed(source, tmp_path):
    dest = tmp_path / "published"
    run("publish", source, dest, "--base-url", BASE)
    (source / "small.png").write_bytes(b"no longer a PNG")
    assert run("publish", source, dest, "--base-url", BASE) == 1
    assert not (dest / "_publish" / "state" / "small.json").exists()
    assert run("manifest", dest, "--source", source) == 0
    labels = [c["id"] for c in read_json(dest / "manifests" / "_root.json")["items"]]
    assert f"{BASE}/canvas/small" not in labels


def test_metadata_file_that_is_not_utf8_is_a_usage_error(source, tmp_path, capsys):
    (source / "metadata.csv").write_bytes("file_name,label\nsmall.png,Caf\xe9\n".encode("latin-1"))
    assert run("publish", source, tmp_path / "published", "--base-url", BASE) == 2
    assert "UTF-8" in capsys.readouterr().err


def test_https_rights_uri_is_stored_in_the_http_form(source, tmp_path):
    dest = tmp_path / "published"
    run("publish", source, dest, "--base-url", BASE, "--rights", "https://creativecommons.org/publicdomain/zero/1.0/")
    assert read_json(dest / "collection.json")["rights"] == "http://creativecommons.org/publicdomain/zero/1.0/"


class FakeBucketFs:
    """A stand-in for HfFileSystem on one bucket, with the same folder behaviour."""

    def __init__(self, bucket, paths):
        self.root = f"buckets/{bucket}"
        self.paths = [f"{self.root}/{path}" for path in paths]

    def invalidate_cache(self):
        pass

    def find(self, root, detail=True):
        return {
            path: {"name": path, "type": "file", "size": 10, "xet_hash": "h"}
            for path in self.paths
            if path == root or path.startswith(root + "/")
        }

    def ls(self, root, detail=True):
        children = {path[len(root) + 1 :].split("/")[0] for path in self.paths if path.startswith(root + "/")}
        if not children:
            raise FileNotFoundError(root)
        folders = {c for c in children if any(p.startswith(f"{root}/{c}/") for p in self.paths)}
        return [{"name": f"{root}/{c}", "type": "directory" if c in folders else "file"} for c in sorted(children)]


def test_bucket_source_lists_only_its_own_folder():
    fs = FakeBucketFs(
        "o/b", ["scans/a.jpg", "scans/book/b.jpg", "scans/_publish/x.json", "scans-old/z.jpg", "scansx.jpg"]
    )
    listed = [f.rel_path for f in recipe.BucketSource(None, "o/b", "scans", fs=fs).list_files()]
    assert listed == ["a.jpg", "book/b.jpg"]
    whole_bucket = [f.rel_path for f in recipe.BucketSource(None, "o/b", "", fs=fs).list_files()]
    assert "scans-old/z.jpg" in whole_bucket and "scans/_publish/x.json" not in whole_bucket


def test_bucket_destination_paths_are_relative_to_its_prefix():
    files = [
        "pub/images/page1/info.json",
        "pub/images/page10/info.json",
        "pub/collection.json",
        "published/images/page1/x",
    ]
    dest = recipe.BucketDest(None, "o/b", "pub", fs=FakeBucketFs("o/b", files))
    assert dest._list("images/page1") == ["pub/images/page1/info.json"]
    assert dest._list("collection.json") == ["pub/collection.json"]
    assert dest.list_names("images") == ["images/page1/info.json", "images/page10/info.json"]
    assert dest.list_dirs("images") == {"page1", "page10"}
    assert dest.list_dirs("nothing-here") == set()
    root = recipe.BucketDest(None, "o/b", "", fs=FakeBucketFs("o/b", files))
    assert root.list_dirs("") == {"pub", "published"}
    assert root.list_names("pub/images/page1") == ["pub/images/page1/info.json"]


def test_bucket_prefixes_that_only_share_text_do_not_overlap():
    source = recipe.Location(kind="bucket", bucket="o/b", prefix="pub")
    recipe.check_no_overlap(source, recipe.Location(kind="bucket", bucket="o/b", prefix="published"))
    with pytest.raises(recipe.UserError):
        recipe.check_no_overlap(source, recipe.Location(kind="bucket", bucket="o/b", prefix="pub/out"))


def test_jsonl_line_that_is_not_an_object_is_a_usage_error(source, tmp_path, capsys):
    (source / "metadata.csv").unlink()
    (source / "metadata.jsonl").write_text("[]\n")
    assert run("publish", source, tmp_path / "published", "--base-url", BASE) == 2
    assert "JSON object" in capsys.readouterr().err


def test_first_run_where_every_image_fails_writes_nothing(tmp_path, capsys):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.jpg").write_bytes(b"not an image")
    dest = tmp_path / "published"
    assert run("publish", source, dest, "--base-url", BASE) == 1
    assert not (dest / "collection.json").exists()
    assert "FAILED" in capsys.readouterr().out


def test_check_keeps_other_problems_when_one_image_cannot_be_checked(source, tmp_path, monkeypatch, capsys):
    dest = tmp_path / "published"
    run("publish", source, dest, "--base-url", BASE)
    (dest / "images" / "rotated" / "full" / "max" / "0" / "default.jpg").unlink()
    real_exists = recipe.LocalReader.exists

    def flaky(self, path):
        if path.startswith("images/small/"):
            raise recipe.UserError("Network error while reading " + path)
        return real_exists(self, path)

    monkeypatch.setattr(recipe.LocalReader, "exists", flaky)
    capsys.readouterr()
    assert run("check", dest, "--json") == 1
    problems = " ".join(json.loads(capsys.readouterr().out)["problems"])
    assert "canvas/small: could not finish the check" in problems
    assert "canvas/rotated: full image cannot be fetched" in problems


def test_sparse_metadata_gives_nulls_and_odd_column_names_are_kept(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("a", "b", "c"):
        picture(200, 100).jpegsave(str(source / f"{name}.jpg"))
    (source / "metadata.csv").write_text("file_name,width,Date,date\na.jpg,wide,1900,\nb.jpg,,,1901\n")
    dest = tmp_path / "published"
    assert run("publish", source, dest, "--base-url", BASE) == 0
    rows = {r["image_id"]: r for r in pq.read_table(io.BytesIO((dest / "index.parquet").read_bytes())).to_pylist()}
    assert rows["a"]["width"] == 200 and rows["a"]["meta_width"] == "wide"
    assert (rows["a"]["meta_Date"], rows["a"]["meta_date"]) == ("1900", None)
    assert (rows["b"]["meta_Date"], rows["b"]["meta_date"]) == (None, "1901")
    assert rows["c"]["meta_width"] is None and rows["c"]["meta_date"] is None


def test_check_reports_a_missing_derivative(source, tmp_path, capsys):
    dest = tmp_path / "published"
    run("publish", source, dest, "--base-url", BASE)
    (dest / "images" / "book-a__page2" / "full" / "1024,751" / "0" / "default.jpg").unlink()
    capsys.readouterr()
    assert run("check", dest, "--json") == 1
    assert "derivative image cannot be fetched" in " ".join(json.loads(capsys.readouterr().out)["problems"])


def test_state_file_is_removed_before_the_image_folder_is_cleared(source, tmp_path, monkeypatch):
    dest = tmp_path / "published"
    run("publish", source, dest, "--base-url", BASE)
    state_dir = dest / "_publish" / "state"
    assert len(list(state_dir.iterdir())) == 5

    def killed(self, dest_prefix):
        raise SystemExit("the process stops while it replaces an image")

    monkeypatch.setattr(recipe.LocalDest, "clear", killed)
    with pytest.raises(SystemExit):
        run("publish", source, dest, "--base-url", BASE, "--force", "--workers", "1")
    # Every image whose folder was about to be cleared has lost its state file,
    # so the next run cannot treat a half-written folder as complete.
    assert list(state_dir.iterdir()) == []
    monkeypatch.undo()
    assert run("publish", source, dest, "--base-url", BASE) == 0
    assert run("check", dest) == 0
