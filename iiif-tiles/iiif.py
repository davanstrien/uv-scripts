# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "pyvips[binary]",
#     "huggingface-hub>=1.22",
#     "pyarrow",
# ]
# ///
"""Publish a folder or bucket of images as a static IIIF collection.

Writes IIIF Image API 3 level 0 files, Presentation 3 manifests, a Collection and a
Parquet index. Everything is static, so a Hugging Face Storage Bucket can serve it.

Usage:
    uv run iiif.py publish ./scans hf://buckets/org/my-collection --space org/my-collection
    uv run iiif.py manifest hf://buckets/org/my-collection --source ./scans
    uv run iiif.py check hf://buckets/org/my-collection

Hugging Face Jobs:
    hf jobs uv run \
      https://huggingface.co/datasets/uv-scripts/iiif-tiles/raw/main/iiif.py \
      publish hf://buckets/org/source-images hf://buckets/org/my-collection
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import sys
import tempfile
import traceback
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

TOOL_VERSION = "1.0.0"
SCHEMA_VERSION = 1

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".webp"}
METADATA_FILENAMES = ("metadata.csv", "metadata.jsonl")
ITEMS_FILENAME = "items.csv"

THUMBNAIL_EDGE = 400  # long edge of the canvas thumbnail
DERIVATIVE_EDGE = 1024  # long edge of the model-sized image
ROOT_ITEM_ID = "_root"  # item id for images that are not in a subfolder
ID_SEPARATOR = "__"  # joins folder and file slugs; a slug never contains "_"

RESERVED_IMAGE_COLUMNS = {"file_name", "label", "rights", "attribution"}
RESERVED_ITEM_COLUMNS = {"folder", "label", "rights", "attribution"}
RIGHTS_PREFIXES = (
    "http://creativecommons.org/",
    "https://creativecommons.org/",
    "http://rightsstatements.org/",
    "https://rightsstatements.org/",
)

PRESENTATION_CONTEXT = "http://iiif.io/api/presentation/3/context.json"
IMAGE_CONTEXT = "http://iiif.io/api/image/3/context.json"
VIEWER_VERSION = "1.3.2"  # triiiceratops release pinned in the viewer Space
UPLOAD_CHUNK = 500  # files per bucket batch call


class UserError(Exception):
    """A problem the user can fix. `hint` names the next thing to do."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.message = message
        self.hint = hint


class Reporter:
    """Progress goes to stderr, so `--json` leaves stdout for the result only."""

    def __init__(self, json_mode: bool):
        self.json_mode = json_mode

    def info(self, text: str) -> None:
        print(text, file=sys.stderr if self.json_mode else sys.stdout, flush=True)

    def warn(self, text: str) -> None:
        print(f"warning: {text}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------
# Ids
# --------------------------------------------------------------------------------------


def slugify(text: str) -> str:
    """Return a lowercase ASCII slug that uses only a-z, 0-9 and "-".

    If transliteration loses a letter or digit (for example a CJK name), a short hash of
    the original text is added, so different names do not fall onto the same slug.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    lossy = any(ord(ch) > 127 and ch.isalnum() for ch in without_marks)
    ascii_text = without_marks.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    if lossy or not slug:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        slug = f"{slug}-{digest}" if slug else f"u-{digest}"
    return slug


def item_id_for(folder: str) -> str:
    """Item id for a source folder. `folder` is "" for images in the source root."""
    if folder == "":
        return ROOT_ITEM_ID
    return ID_SEPARATOR.join(slugify(part) for part in PurePosixPath(folder).parts)


def image_id_for(rel_path: str) -> str:
    """Image id for a source path: the folder slugs and the file stem slug, joined."""
    path = PurePosixPath(rel_path)
    parts = list(path.parent.parts) + [path.stem]
    return ID_SEPARATOR.join(slugify(part) for part in parts)


def natural_key(text: str) -> list:
    """Sort key that puts page2 before page10."""
    return [int(piece) if piece.isdigit() else piece.lower() for piece in re.split(r"(\d+)", text)]


# --------------------------------------------------------------------------------------
# Locations: where images come from and where the IIIF files go
# --------------------------------------------------------------------------------------


@dataclass
class Location:
    kind: str  # "local" or "bucket"
    path: str = ""  # local directory
    bucket: str = ""  # "org/name"
    prefix: str = ""  # folder inside the bucket, without slashes at the ends


def parse_location(text: str, role: str) -> Location:
    if text.startswith("hf://buckets/"):
        parts = text[len("hf://buckets/") :].strip("/").split("/")
        if len(parts) < 2:
            raise UserError(f"{role} bucket must be hf://buckets/<owner>/<name>, got {text}")
        return Location(kind="bucket", bucket="/".join(parts[:2]), prefix="/".join(parts[2:]))
    if text.startswith("hf://datasets/"):
        raise UserError(
            "Hub datasets are not supported as a location yet.",
            "Copy the image files to a folder or a bucket, then run publish on that.",
        )
    if "://" in text:
        raise UserError(
            f"Unsupported {role} location: {text}",
            "Use a local folder or hf://buckets/<owner>/<name>. "
            "For images in other cloud storage, copy them into a Hugging Face bucket first.",
        )
    return Location(kind="local", path=text)


@dataclass
class SourceFile:
    rel_path: str  # POSIX path relative to the source root
    size: int
    fingerprint: str  # changes when the file content changes
    handle: object = None  # local Path or BucketFile


class LocalSource:
    def __init__(self, root: str):
        self.root = Path(root)
        if not self.root.is_dir():
            raise UserError(f"Source folder does not exist: {root}")

    def list_files(self) -> list[SourceFile]:
        files = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or any(part.startswith(".") for part in path.relative_to(self.root).parts):
                continue
            stat = path.stat()
            rel = path.relative_to(self.root).as_posix()
            files.append(SourceFile(rel, stat.st_size, f"{stat.st_size}-{stat.st_mtime_ns}", path))
        return files

    def read_text(self, source_file: SourceFile) -> str:
        return Path(source_file.handle).read_text(encoding="utf-8-sig")

    def fetch(self, source_file: SourceFile, work_dir: Path) -> Path:
        return Path(source_file.handle)


class BucketSource:
    def __init__(self, api, bucket: str, prefix: str):
        self.api = api
        self.bucket = bucket
        self.prefix = prefix

    def list_files(self) -> list[SourceFile]:
        files = []
        strip = len(self.prefix) + 1 if self.prefix else 0
        for entry in self.api.list_bucket_tree(self.bucket, prefix=self.prefix or None, recursive=True):
            if getattr(entry, "type", "file") != "file" or not hasattr(entry, "size"):
                continue
            rel = entry.path[strip:]
            if not rel or any(part.startswith((".", "_publish")) for part in rel.split("/")):
                continue
            marker = getattr(entry, "xet_hash", None) or getattr(entry, "mtime", None) or ""
            files.append(SourceFile(rel, entry.size, f"{entry.size}-{marker}", entry))
        files.sort(key=lambda f: f.rel_path)
        return files

    def read_text(self, source_file: SourceFile) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "file"
            self.api.download_bucket_files(self.bucket, files=[(source_file.handle, str(target))])
            return target.read_text(encoding="utf-8-sig")

    def fetch(self, source_file: SourceFile, work_dir: Path) -> Path:
        target = work_dir / ("source" + PurePosixPath(source_file.rel_path).suffix)
        self.api.download_bucket_files(self.bucket, files=[(source_file.handle, str(target))])
        return target


class LocalDest:
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def describe(self) -> str:
        return str(self.root)

    def put_tree(self, local_dir: Path, dest_prefix: str) -> None:
        shutil.copytree(local_dir, self.root / dest_prefix, dirs_exist_ok=True)

    def put_files(self, files: list[tuple[bytes, str]]) -> None:
        for data, dest_path in files:
            target = self.root / dest_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

    def read_bytes(self, dest_path: str) -> bytes | None:
        target = self.root / dest_path
        return target.read_bytes() if target.is_file() else None

    def read_all(self, dest_prefix: str) -> dict[str, bytes]:
        folder = self.root / dest_prefix
        if not folder.is_dir():
            return {}
        return {p.relative_to(self.root).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()}

    def clear(self, dest_prefix: str) -> None:
        shutil.rmtree(self.root / dest_prefix, ignore_errors=True)

    def remove(self, dest_path: str) -> None:
        (self.root / dest_path).unlink(missing_ok=True)

    def list_dirs(self, dest_prefix: str) -> set[str]:
        """Names of the folders directly inside `dest_prefix`."""
        folder = self.root / dest_prefix
        return {child.name for child in folder.iterdir() if child.is_dir()} if folder.is_dir() else set()


class BucketDest:
    def __init__(self, api, bucket: str, prefix: str):
        self.api = api
        self.bucket = bucket
        self.prefix = prefix

    def describe(self) -> str:
        return f"hf://buckets/{self.bucket}" + (f"/{self.prefix}" if self.prefix else "")

    def _full(self, dest_path: str) -> str:
        return f"{self.prefix}/{dest_path}" if self.prefix else dest_path

    def _list(self, dest_prefix: str) -> list:
        entries = self.api.list_bucket_tree(self.bucket, prefix=self._full(dest_prefix), recursive=True)
        return [e for e in entries if getattr(e, "type", "file") == "file" and hasattr(e, "size")]

    def put_tree(self, local_dir: Path, dest_prefix: str) -> None:
        pairs = []
        for path in sorted(local_dir.rglob("*")):
            if path.is_file():
                pairs.append((str(path), self._full(f"{dest_prefix}/{path.relative_to(local_dir).as_posix()}")))
        for start in range(0, len(pairs), UPLOAD_CHUNK):
            self.api.batch_bucket_files(self.bucket, add=pairs[start : start + UPLOAD_CHUNK])

    def put_files(self, files: list[tuple[bytes, str]]) -> None:
        pairs = [(data, self._full(dest_path)) for data, dest_path in files]
        for start in range(0, len(pairs), UPLOAD_CHUNK):
            self.api.batch_bucket_files(self.bucket, add=pairs[start : start + UPLOAD_CHUNK])

    def read_bytes(self, dest_path: str) -> bytes | None:
        wanted = self._full(dest_path)
        entries = [entry for entry in self._list(dest_path) if entry.path == wanted]
        return self._download(entries).get(dest_path)

    def read_all(self, dest_prefix: str) -> dict[str, bytes]:
        return self._download(self._list(dest_prefix + "/"))

    def _download(self, entries: list) -> dict[str, bytes]:
        if not entries:
            return {}
        strip = len(self.prefix) + 1 if self.prefix else 0
        result = {}
        with tempfile.TemporaryDirectory() as tmp:
            pairs = [(entry, str(Path(tmp) / f"{number}.bin")) for number, entry in enumerate(entries)]
            self.api.download_bucket_files(self.bucket, files=pairs)
            for entry, local in pairs:
                if Path(local).is_file():
                    result[entry.path[strip:]] = Path(local).read_bytes()
        return result

    def clear(self, dest_prefix: str) -> None:
        paths = [entry.path for entry in self._list(dest_prefix + "/")]
        for start in range(0, len(paths), UPLOAD_CHUNK):
            self.api.batch_bucket_files(self.bucket, delete=paths[start : start + UPLOAD_CHUNK])

    def remove(self, dest_path: str) -> None:
        wanted = self._full(dest_path)
        if any(entry.path == wanted for entry in self._list(dest_path)):
            self.api.batch_bucket_files(self.bucket, delete=[wanted])

    def list_dirs(self, dest_prefix: str) -> set[str]:
        """Names of the folders directly inside `dest_prefix`."""
        parent = self._full(dest_prefix) + "/"
        names = set()
        for entry in self.api.list_bucket_tree(self.bucket, prefix=parent, recursive=False):
            if entry.path.startswith(parent) and entry.path != parent:
                names.add(entry.path[len(parent) :].strip("/").split("/")[0])
        return names


def hub_api():
    """Hub client with transfer progress bars off: one bar per file hides this tool's own progress."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    return HfApi()


def open_source(location: Location):
    if location.kind == "local":
        return LocalSource(location.path)
    return BucketSource(hub_api(), location.bucket, location.prefix)


def open_dest(location: Location, create: bool, allow_private: bool):
    if location.kind == "local":
        return LocalDest(location.path)
    from huggingface_hub.errors import HfHubHTTPError

    api = hub_api()
    try:
        info = api.bucket_info(location.bucket)
    except HfHubHTTPError as error:
        if not create:
            raise UserError(f"Cannot open bucket {location.bucket}: {error}") from error
        api.create_bucket(location.bucket, private=False, exist_ok=True)
        info = api.bucket_info(location.bucket)
    if getattr(info, "private", False) and not allow_private:
        raise UserError(
            f"Bucket {location.bucket} is private. IIIF viewers cannot read a private bucket.",
            f"Run: hf buckets settings {location.bucket} --public   (or pass --allow-private)",
        )
    return BucketDest(api, location.bucket, location.prefix)


def check_no_overlap(source: Location, dest: Location) -> None:
    """Refuse a destination inside the source (or the reverse): the next run would
    read the published JPEGs as new source images."""
    if source.kind != dest.kind:
        return
    if source.kind == "local":
        first, second = Path(source.path).resolve(), Path(dest.path).resolve()
        overlap = first == second or first in second.parents or second in first.parents
    else:
        first, second = source.prefix.strip("/"), dest.prefix.strip("/")
        nested = first == second or not first or not second
        nested = nested or first.startswith(second + "/") or second.startswith(first + "/")
        overlap = source.bucket == dest.bucket and nested
    if overlap:
        raise UserError(
            "The source and the destination overlap.",
            "Use a destination outside the source: a different folder, bucket, or bucket prefix.",
        )


def default_base_url(location: Location, base_url: str | None) -> str:
    if base_url:
        return base_url.rstrip("/")
    if location.kind == "bucket":
        url = f"https://huggingface.co/buckets/{location.bucket}/resolve"
        return f"{url}/{location.prefix}" if location.prefix else url
    raise UserError(
        "A local destination needs --base-url: the URL where these files will be served.",
        "For a local test, pass --base-url http://localhost:8000 and serve the folder with: python -m http.server 8000",
    )


# --------------------------------------------------------------------------------------
# Plan: read the source listing and metadata into items and images
# --------------------------------------------------------------------------------------


@dataclass
class PlannedImage:
    id: str
    source: str  # path relative to the source root
    label: str
    rights: str = ""
    attribution: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class PlannedItem:
    id: str
    folder: str  # "" for the source root
    label: str
    rights: str = ""
    attribution: str = ""
    metadata: dict = field(default_factory=dict)
    images: list = field(default_factory=list)


def read_rows(text: str, filename: str) -> list[dict]:
    """Read a metadata file into a list of dicts with string values."""
    rows = []
    if filename.endswith(".jsonl"):
        for line in text.splitlines():
            if line.strip():
                rows.append(json.loads(line))
    else:
        rows = list(csv.DictReader(io.StringIO(text)))
    cleaned = []
    for row in rows:
        cleaned.append({str(k).strip(): ("" if v is None else str(v).strip()) for k, v in row.items() if k})
    return cleaned


def check_rights(value: str, where: str) -> None:
    if value and not value.startswith(RIGHTS_PREFIXES):
        raise UserError(
            f"{where}: rights must be a Creative Commons or RightsStatements.org URI, got '{value}'.",
            "Example: http://creativecommons.org/publicdomain/mark/1.0/ . "
            "Put free-text credit lines in the 'attribution' column.",
        )


def build_plan(source, title: str, reporter: Reporter) -> tuple[list[PlannedItem], dict[str, SourceFile]]:
    """List the source and return the items to publish plus the source file of each image."""
    files = source.list_files()
    by_path = {f.rel_path: f for f in files}
    image_files = [f for f in files if PurePosixPath(f.rel_path).suffix.lower() in IMAGE_EXTENSIONS]
    if not image_files:
        raise UserError(
            "No images found in the source.",
            f"Supported extensions: {', '.join(sorted(IMAGE_EXTENSIONS))}",
        )

    # Image ids must be unique. Two source paths with one id is an error, never a silent overwrite.
    source_files: dict[str, SourceFile] = {}
    path_of_id: dict[str, str] = {}
    for source_file in image_files:
        image_id = image_id_for(source_file.rel_path)
        if image_id in path_of_id:
            raise UserError(
                f"Two files produce the same image id '{image_id}': "
                f"'{path_of_id[image_id]}' and '{source_file.rel_path}'.",
                "Rename one of them. Ids ignore case, punctuation and the file extension.",
            )
        path_of_id[image_id] = source_file.rel_path
        source_files[image_id] = source_file

    # Per-image metadata. A metadata file applies to paths relative to its own folder.
    image_rows: dict[str, dict] = {}
    for source_file in files:
        path = PurePosixPath(source_file.rel_path)
        if path.name not in METADATA_FILENAMES:
            continue
        rows = read_rows(source.read_text(source_file), path.name)
        folder = "" if str(path.parent) == "." else str(path.parent)
        unmatched = []
        for row in rows:
            file_name = row.get("file_name", "")
            target = f"{folder}/{file_name}" if folder else file_name
            if target in by_path and image_id_for(target) in source_files:
                image_rows[target] = row
            else:
                unmatched.append(file_name)
        if rows and "file_name" not in rows[0]:
            raise UserError(f"{source_file.rel_path} has no 'file_name' column.")
        if unmatched:
            shown = ", ".join(unmatched[:5])
            reporter.warn(
                f"{source_file.rel_path}: {len(unmatched)} row(s) match no image file "
                f"(first: {shown}). These rows are ignored."
            )

    # Per-item metadata from items.csv in the source root.
    item_rows: dict[str, dict] = {}
    if ITEMS_FILENAME in by_path:
        rows = read_rows(source.read_text(by_path[ITEMS_FILENAME]), ITEMS_FILENAME)
        if rows and "folder" not in rows[0]:
            raise UserError(f"{ITEMS_FILENAME} has no 'folder' column.")
        for row in rows:
            item_rows[row.get("folder", "").strip("/")] = row

    items: dict[str, PlannedItem] = {}
    folder_of_item: dict[str, str] = {}
    for source_file in sorted(image_files, key=lambda f: natural_key(f.rel_path)):
        path = PurePosixPath(source_file.rel_path)
        folder = "" if str(path.parent) == "." else str(path.parent)
        item_id = item_id_for(folder)
        if folder_of_item.setdefault(item_id, folder) != folder:
            raise UserError(
                f"Two folders produce the same item id '{item_id}': '{folder_of_item[item_id]}' and '{folder}'.",
                "Rename one of them.",
            )
        if item_id not in items:
            row = item_rows.get(folder, {})
            check_rights(row.get("rights", ""), f"{ITEMS_FILENAME} folder '{folder}'")
            items[item_id] = PlannedItem(
                id=item_id,
                folder=folder,
                label=row.get("label") or (folder if folder else title),
                rights=row.get("rights", ""),
                attribution=row.get("attribution", ""),
                metadata={k: v for k, v in row.items() if k not in RESERVED_ITEM_COLUMNS and v},
            )
        row = image_rows.get(source_file.rel_path, {})
        check_rights(row.get("rights", ""), f"metadata for '{source_file.rel_path}'")
        items[item_id].images.append(
            PlannedImage(
                id=image_id_for(source_file.rel_path),
                source=source_file.rel_path,
                label=row.get("label") or path.stem,
                rights=row.get("rights", ""),
                attribution=row.get("attribution", ""),
                metadata={k: v for k, v in row.items() if k not in RESERVED_IMAGE_COLUMNS and v},
            )
        )

    for folder in item_rows:
        if item_id_for(folder) not in items or items[item_id_for(folder)].folder != folder:
            reporter.warn(f"{ITEMS_FILENAME}: folder '{folder}' has no images. This row is ignored.")

    ordered = sorted(items.values(), key=lambda item: natural_key(item.folder))
    return ordered, source_files


# --------------------------------------------------------------------------------------
# Image processing
# --------------------------------------------------------------------------------------


@dataclass
class Settings:
    tile_size: int = 512
    tile_threshold: int = 2048
    quality: int = 85

    def key(self) -> str:
        return (
            f"v{SCHEMA_VERSION}-t{self.tile_size}-n{self.tile_threshold}-q{self.quality}"
            f"-th{THUMBNAIL_EDGE}-d{DERIVATIVE_EDGE}"
        )


def fit_long_edge(width: int, height: int, edge: int) -> tuple[int, int] | None:
    """Size with the long edge equal to `edge`, or None if the image is not larger.

    The short edge is rounded half up. This rule is part of the published URLs.
    """
    if max(width, height) <= edge:
        return None
    if width >= height:
        return edge, max(1, (height * edge * 2 + width) // (2 * width))
    return max(1, (width * edge * 2 + height) // (2 * height)), edge


def normalise(image):
    """Return an upright 8-bit sRGB or greyscale image without alpha, plus notes on changes."""
    notes = []
    orientation = image.get("orientation") if image.get_typeof("orientation") else 1
    if orientation != 1:
        image = image.autorot()
        notes.append(f"rotated (EXIF orientation {orientation})")
    if image.interpretation == "cmyk":
        notes.append("converted from CMYK")
    if image.format != "uchar":
        notes.append(f"converted from {image.format} samples to 8-bit")
    if image.interpretation in ("b-w", "grey16"):
        image = image.colourspace("b-w")
    else:
        has_profile = image.get_typeof("icc-profile-data") != 0
        converted = None
        if has_profile and image.interpretation == "cmyk":
            try:
                converted = image.icc_transform("srgb")
            except Exception:  # noqa: BLE001 - a broken embedded profile; use the built-in conversion
                converted = None
        image = converted if converted is not None else image.colourspace("srgb")
    if image.format != "uchar":
        image = image.cast("uchar")
    if image.hasalpha():
        image = image.flatten(background=255)
        notes.append("alpha flattened onto white")
    return image, notes


def process_image(source_path: Path, out_dir: Path, settings: Settings) -> dict:
    """Write every IIIF image file for one source image into `out_dir`.

    Returns the facts that the state file records. `info.json` is written later, because
    it contains the base URL.
    """
    import pyvips

    image = pyvips.Image.new_from_file(str(source_path))
    notes = []
    pages = image.get("n-pages") if image.get_typeof("n-pages") else 1
    if pages > 1:
        second = pyvips.Image.new_from_file(str(source_path), page=1)
        if (second.width, second.height) == (image.width, image.height):
            notes.append(f"multi-page file: only page 1 of {pages} is published")
    image, more_notes = normalise(image)
    notes.extend(more_notes)
    width, height = image.width, image.height
    if width * height * image.bands < 400_000_000:
        image = image.copy_memory()  # decode once, then write several outputs

    save = {"Q": settings.quality, "keep": "none"}
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_full(name: str, picture) -> None:
        target = out_dir / "full" / name / "0" / "default.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        picture.jpegsave(str(target), **save)

    sizes = set()
    scale_factors = []
    tiled = max(width, height) > settings.tile_threshold
    if tiled:
        with tempfile.TemporaryDirectory() as tmp:
            image.dzsave(
                str(Path(tmp) / "dz"),
                layout="iiif3",
                tile_size=settings.tile_size,
                overlap=0,
                suffix=f".jpg[Q={settings.quality},keep=none]",
            )
            dz_dir = Path(tmp) / "dz"
            scale_factors = json.loads((dz_dir / "info.json").read_text())["tiles"][0]["scaleFactors"]
            for child in dz_dir.iterdir():
                if child.is_dir():
                    shutil.copytree(child, out_dir / child.name, dirs_exist_ok=True)
        for folder in (out_dir / "full").iterdir():
            match = re.fullmatch(r"(\d+),(\d+)", folder.name)
            if match:
                sizes.add((int(match.group(1)), int(match.group(2))))
    else:
        # A small image has no tiles. Viewers ask for it by its exact size.
        write_full(f"{width},{height}", image)
        sizes.add((width, height))

    write_full("max", image)

    thumbnail = fit_long_edge(width, height, THUMBNAIL_EDGE) or (width, height)
    derivative = fit_long_edge(width, height, DERIVATIVE_EDGE) or (width, height)
    # The thumbnail and the derivative are extra files. Manifests and the index give their
    # URLs. They are not listed in info.json `sizes`: OpenSeadragon reads every entry there
    # as a tile level, and asks for tiles that do not exist when an entry is not one.
    for size in (thumbnail, derivative):
        if size != (width, height) and size not in sizes:
            write_full(f"{size[0]},{size[1]}", image.thumbnail_image(size[0], height=size[1], size="force"))

    file_count = sum(1 for p in out_dir.rglob("*") if p.is_file())
    return {
        "width": width,
        "height": height,
        "tiled": tiled,
        "tile_size": settings.tile_size if tiled else None,
        "scale_factors": scale_factors,
        "sizes": [list(size) for size in sorted(sizes)],
        "thumbnail": list(thumbnail),
        "derivative": list(derivative),
        "notes": notes,
        "files": file_count + 1,  # plus info.json
    }


# --------------------------------------------------------------------------------------
# IIIF documents
# --------------------------------------------------------------------------------------


def image_urls(base_url: str, state: dict) -> dict:
    service = f"{base_url}/images/{state['id']}"
    width, height = state["width"], state["height"]

    def sized(size: list) -> str:
        if tuple(size) == (width, height):
            return f"{service}/full/max/0/default.jpg"
        return f"{service}/full/{size[0]},{size[1]}/0/default.jpg"

    return {
        "service": service,
        "canvas": f"{base_url}/canvas/{state['id']}",
        "full": f"{service}/full/max/0/default.jpg",
        "thumbnail": sized(state["thumbnail"]),
        "derivative": sized(state["derivative"]),
    }


def info_json(base_url: str, state: dict) -> dict:
    info = {
        "@context": IMAGE_CONTEXT,
        "id": f"{base_url}/images/{state['id']}",
        "type": "ImageService3",
        "protocol": "http://iiif.io/api/image",
        "profile": "level0",
        "width": state["width"],
        "height": state["height"],
        "sizes": [{"width": w, "height": h} for w, h in state["sizes"]],
    }
    if state["tiled"]:
        info["tiles"] = [{"width": state["tile_size"], "scaleFactors": state["scale_factors"]}]
    else:
        # One tile that covers the whole image. Viewers then ask for full/max, which exists.
        # Without it, OpenSeadragon builds "full/w,/0/default.jpg" URLs from `sizes`,
        # and a static level 0 folder does not have those.
        # The tile is square because some viewers ignore a tile "height".
        info["tiles"] = [{"width": max(state["width"], state["height"]), "scaleFactors": [1]}]
    return info


def text_value(text: str, language: str) -> dict:
    return {language: [text]}


def metadata_pairs(metadata: dict, language: str) -> list:
    return [{"label": text_value(k, language), "value": text_value(v, language)} for k, v in metadata.items()]


def add_descriptive(resource: dict, rights: str, attribution: str, metadata: dict, language: str) -> None:
    if metadata:
        resource["metadata"] = metadata_pairs(metadata, language)
    if rights:
        resource["rights"] = rights
    if attribution:
        resource["requiredStatement"] = {
            "label": {"en": ["Attribution"]},
            "value": text_value(attribution, language),
        }


def thumbnail_resource(base_url: str, state: dict) -> list:
    return [
        {
            "id": image_urls(base_url, state)["thumbnail"],
            "type": "Image",
            "format": "image/jpeg",
            "width": state["thumbnail"][0],
            "height": state["thumbnail"][1],
        }
    ]


def canvas_json(base_url: str, image: PlannedImage, state: dict, language: str) -> dict:
    urls = image_urls(base_url, state)
    canvas = {
        "id": urls["canvas"],
        "type": "Canvas",
        "label": text_value(image.label, language),
        "width": state["width"],
        "height": state["height"],
        "thumbnail": thumbnail_resource(base_url, state),
    }
    add_descriptive(canvas, image.rights, image.attribution, image.metadata, language)
    canvas["items"] = [
        {
            "id": f"{urls['canvas']}/page",
            "type": "AnnotationPage",
            "items": [
                {
                    "id": f"{urls['canvas']}/annotation",
                    "type": "Annotation",
                    "motivation": "painting",
                    "body": {
                        "id": urls["full"],
                        "type": "Image",
                        "format": "image/jpeg",
                        "width": state["width"],
                        "height": state["height"],
                        "service": [{"id": urls["service"], "type": "ImageService3", "profile": "level0"}],
                    },
                    "target": urls["canvas"],
                }
            ],
        }
    ]
    return canvas


def manifest_json(base_url: str, item: PlannedItem, states: dict, config: dict) -> dict:
    language = config["language"]
    manifest = {
        "@context": PRESENTATION_CONTEXT,
        "id": f"{base_url}/manifests/{item.id}.json",
        "type": "Manifest",
        "label": text_value(item.label, language),
    }
    add_descriptive(
        manifest,
        item.rights or config["rights"],
        item.attribution or config["attribution"],
        item.metadata,
        language,
    )
    manifest["thumbnail"] = thumbnail_resource(base_url, states[item.images[0].id])
    manifest["seeAlso"] = [
        {
            "id": f"{base_url}/index.parquet",
            "type": "Dataset",
            "label": {"en": ["Image index (Parquet): one row per image, with URLs and metadata"]},
            "format": "application/vnd.apache.parquet",
        }
    ]
    manifest["partOf"] = [{"id": f"{base_url}/collection.json", "type": "Collection"}]
    manifest["items"] = [canvas_json(base_url, image, states[image.id], language) for image in item.images]
    return manifest


def collection_json(base_url: str, items: list[PlannedItem], states: dict, config: dict) -> dict:
    language = config["language"]
    collection = {
        "@context": PRESENTATION_CONTEXT,
        "id": f"{base_url}/collection.json",
        "type": "Collection",
        "label": text_value(config["title"], language),
    }
    add_descriptive(collection, config["rights"], config["attribution"], {}, language)
    collection["items"] = [
        {
            "id": f"{base_url}/manifests/{item.id}.json",
            "type": "Manifest",
            "label": text_value(item.label, language),
            "thumbnail": thumbnail_resource(base_url, states[item.images[0].id]),
        }
        for item in items
    ]
    return collection


def index_parquet(base_url: str, items: list[PlannedItem], states: dict, config: dict) -> bytes:
    """One row per image. This table is the interface for programs: only add columns."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    meta_columns = []
    for item in items:
        for image in item.images:
            for column in image.metadata:
                if column not in meta_columns:
                    meta_columns.append(column)
    rows = []
    for item in items:
        for position, image in enumerate(item.images):
            state = states[image.id]
            urls = image_urls(base_url, state)
            row = {
                "image_id": image.id,
                "item_id": item.id,
                "item_label": item.label,
                "position": position,
                "label": image.label,
                "source_path": image.source,
                "width": state["width"],
                "height": state["height"],
                "tiled": state["tiled"],
                "canvas_id": urls["canvas"],
                "manifest_url": f"{base_url}/manifests/{item.id}.json",
                "image_service": urls["service"],
                "image_url": urls["full"],
                "derivative_url": urls["derivative"],
                "derivative_width": state["derivative"][0],
                "derivative_height": state["derivative"][1],
                "thumbnail_url": urls["thumbnail"],
                "rights": image.rights or item.rights or config["rights"],
                "attribution": image.attribution or item.attribution or config["attribution"],
            }
            for column in meta_columns:
                row[f"meta_{column}"] = image.metadata.get(column)
            rows.append(row)
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buffer)
    return buffer.getvalue()


# --------------------------------------------------------------------------------------
# Catalog: what the last run published. `manifest` can rebuild from it without the source.
# --------------------------------------------------------------------------------------

CATALOG_PATH = "_publish/catalog.json"


def state_path(image_id: str) -> str:
    return f"_publish/state/{image_id}.json"


def to_json_bytes(document: dict) -> bytes:
    return (json.dumps(document, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def catalog_document(items: list[PlannedItem], config: dict) -> dict:
    catalog_items = []
    for item in items:
        entry = {k: getattr(item, k) for k in ("id", "folder", "label", "rights", "attribution", "metadata")}
        entry["images"] = [
            {k: getattr(image, k) for k in ("id", "source", "label", "rights", "attribution", "metadata")}
            for image in item.images
        ]
        catalog_items.append(entry)
    return {"schema_version": SCHEMA_VERSION, "tool_version": TOOL_VERSION, "config": config, "items": catalog_items}


def items_from_catalog(catalog: dict) -> list[PlannedItem]:
    items = []
    for entry in catalog["items"]:
        images = [PlannedImage(**image) for image in entry["images"]]
        fields = {k: v for k, v in entry.items() if k != "images"}
        items.append(PlannedItem(images=images, **fields))
    return items


def load_catalog(dest) -> dict | None:
    data = dest.read_bytes(CATALOG_PATH)
    return json.loads(data) if data else None


def load_states(dest) -> dict[str, dict]:
    states = {}
    for data in dest.read_all("_publish/state").values():
        state = json.loads(data)
        states[state["id"]] = state
    return states


def write_outputs(dest, base_url: str, items: list[PlannedItem], states: dict, config: dict, reporter) -> dict:
    """Write info.json files, manifests, the Collection, the index and the catalog."""
    published = []
    not_tiled = []
    for item in items:
        ready = [image for image in item.images if image.id in states]
        not_tiled.extend(image.source for image in item.images if image.id not in states)
        if ready:
            published.append(
                PlannedItem(item.id, item.folder, item.label, item.rights, item.attribution, item.metadata, ready)
            )
    if not published:
        raise UserError("No image is ready to publish.", "Run publish first and check the errors it reports.")

    files = []
    for item in published:
        for image in item.images:
            files.append((to_json_bytes(info_json(base_url, states[image.id])), f"images/{image.id}/info.json"))
    for item in published:
        files.append((to_json_bytes(manifest_json(base_url, item, states, config)), f"manifests/{item.id}.json"))
    files.append((to_json_bytes(collection_json(base_url, published, states, config)), "collection.json"))
    files.append((index_parquet(base_url, published, states, config), "index.parquet"))
    config = dict(config, base_url=base_url)
    files.append((to_json_bytes(catalog_document(published, config)), CATALOG_PATH))
    reporter.info(f"Writing {len(files)} description files (info.json, manifests, collection, index)")
    dest.put_files(files)

    planned_ids = {image.id for item in published for image in item.images}
    return {
        "items": len(published),
        "images": len(planned_ids),
        "not_tiled": not_tiled,
        "orphans": sorted(set(states) - planned_ids),
    }


# --------------------------------------------------------------------------------------
# Viewer Space
# --------------------------------------------------------------------------------------

VIEWER_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  html, body { margin: 0; height: 100%; font-family: system-ui, sans-serif; }
  header { display: flex; gap: 1rem; align-items: baseline; padding: 0.5rem 1rem; border-bottom: 1px solid #ddd; }
  header h1 { font-size: 1rem; margin: 0; flex: 1; }
  header a { font-size: 0.85rem; }
  triiiceratops-viewer { display: block; width: 100%; height: calc(100vh - 2.6rem); }
</style>
<script src="https://unpkg.com/triiiceratops@__VIEWER_VERSION__/dist/triiiceratops-element.iife.js"></script>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <a href="__ENTRY_URL__" target="_blank" rel="noopener">IIIF __ENTRY_KIND__</a>
  <a href="__INDEX_URL__" target="_blank" rel="noopener">Index (Parquet)</a>
</header>
<triiiceratops-viewer manifest-id="__ENTRY_URL__" config='{"gallery": {"open": true}}'></triiiceratops-viewer>
</body>
</html>
"""


def html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def entry_point(base_url: str, catalog: dict) -> tuple[str, str]:
    """The URL a viewer opens: the manifest when there is one item, else the Collection."""
    if len(catalog["items"]) == 1:
        return f"{base_url}/manifests/{catalog['items'][0]['id']}.json", "manifest"
    return f"{base_url}/collection.json", "collection"


def space_readme(title: str, entry_url: str, entry_kind: str, base_url: str) -> str:
    # A JSON string is a valid YAML double-quoted scalar, so quotes and backslashes are safe.
    yaml_title = json.dumps(" ".join(title.split())[:60], ensure_ascii=False)
    return (
        "---\n"
        f"title: {yaml_title}\n"
        "emoji: 🖼️\ncolorFrom: gray\ncolorTo: indigo\nsdk: static\npinned: false\n"
        "tags:\n  - iiif\n  - uv-script\n"
        "---\n\n"
        f"IIIF viewer for **{' '.join(title.split())}**.\n\n"
        f"- IIIF {entry_kind}: {entry_url}\n"
        f"- Image index (Parquet): {base_url}/index.parquet\n\n"
        "Made with [uv-scripts/iiif-tiles](https://huggingface.co/datasets/uv-scripts/iiif-tiles).\n"
    )


def publish_space(space_id: str, base_url: str, catalog: dict, private: bool) -> str:
    title = catalog["config"]["title"]
    entry_url, entry_kind = entry_point(base_url, catalog)
    html = VIEWER_HTML
    replacements = {
        "__TITLE__": html_escape(title),
        "__VIEWER_VERSION__": VIEWER_VERSION,
        "__ENTRY_URL__": html_escape(entry_url),
        "__ENTRY_KIND__": entry_kind,
        "__INDEX_URL__": html_escape(f"{base_url}/index.parquet"),
    }
    for marker, value in replacements.items():
        html = html.replace(marker, value)
    readme = space_readme(title, entry_url, entry_kind, base_url)
    api = hub_api()
    api.create_repo(space_id, repo_type="space", space_sdk="static", private=private, exist_ok=True)
    for name, text in (("index.html", html), ("README.md", readme)):
        api.upload_file(
            path_or_fileobj=text.encode("utf-8"),
            path_in_repo=name,
            repo_id=space_id,
            repo_type="space",
            commit_message=f"Update {name} (uv-scripts/iiif-tiles)",
        )
    return f"https://huggingface.co/spaces/{space_id}"


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------


def make_config(args, previous: dict | None) -> dict:
    """Command-line values win; otherwise keep what the last run stored."""
    old = (previous or {}).get("config", {})
    config = {
        "title": args.title or old.get("title") or "IIIF Collection",
        "rights": args.rights if args.rights is not None else old.get("rights", ""),
        "attribution": args.attribution if args.attribution is not None else old.get("attribution", ""),
        "language": args.language or old.get("language") or "none",
    }
    check_rights(config["rights"], "--rights")
    return config


def summary_links(base_url: str, catalog: dict) -> dict:
    entry_url, entry_kind = entry_point(base_url, catalog)
    return {
        "entry_url": entry_url,
        "entry_kind": entry_kind,
        "collection_url": f"{base_url}/collection.json",
        "index_url": f"{base_url}/index.parquet",
    }


def command_publish(args, reporter: Reporter) -> dict:
    source_location = parse_location(args.source, "source")
    dest_location = parse_location(args.dest, "destination")
    check_no_overlap(source_location, dest_location)
    base_url = default_base_url(dest_location, args.base_url)
    settings = Settings(args.tile_size, args.tile_threshold, args.quality)

    source = open_source(source_location)
    reporter.info(f"Reading source: {args.source}")
    dest = open_dest(dest_location, create=True, allow_private=args.allow_private)
    previous = load_catalog(dest)
    config = make_config(args, previous)
    items, source_files = build_plan(source, config["title"], reporter)
    total = sum(len(item.images) for item in items)
    reporter.info(f"Found {total} image(s) in {len(items)} item(s)")

    states = load_states(dest)
    # A folder without a state file is left over from a run that was stopped. It is cleared
    # before the image is written again, so files from other settings cannot remain.
    existing_folders = dest.list_dirs("images")
    todo = []
    for item in items:
        for image in item.images:
            state = states.get(image.id)
            fingerprint = source_files[image.id].fingerprint
            current = (
                state is not None
                and not args.force
                and state.get("source_fingerprint") == fingerprint
                and state.get("settings") == settings.key()
            )
            if not current:
                todo.append((image, state is not None or image.id in existing_folders))
    reporter.info(f"{total - len(todo)} image(s) are up to date, {len(todo)} to process")

    failures = []

    def work(job) -> dict:
        image, replace = job
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            local = source.fetch(source_files[image.id], work_dir)
            facts = process_image(local, work_dir / "out", settings)
            state = {
                "schema_version": SCHEMA_VERSION,
                "tool_version": TOOL_VERSION,
                "id": image.id,
                "source": image.source,
                "source_fingerprint": source_files[image.id].fingerprint,
                "settings": settings.key(),
                "quality": settings.quality,
            }
            state.update(facts)
            (work_dir / "out" / "info.json").write_bytes(to_json_bytes(info_json(base_url, state)))
            if replace:
                # Remove the state file first: without it, a failure below cannot leave
                # manifests that point at files which are gone.
                dest.remove(state_path(image.id))
                dest.clear(f"images/{image.id}")
            dest.put_tree(work_dir / "out", f"images/{image.id}")
            # The state file is written last. Its presence means the image is complete.
            dest.put_files([(to_json_bytes(state), state_path(image.id))])
            return state

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(work, job): job for job in todo}
        for future in as_completed(futures):
            image = futures[future][0]
            done += 1
            try:
                state = future.result()
            except Exception as error:  # noqa: BLE001 - one bad image must not stop the run
                failures.append({"source": image.source, "error": f"{type(error).__name__}: {error}"})
                reporter.warn(f"[{done}/{len(todo)}] FAILED {image.source}: {error}")
                states.pop(image.id, None)  # its files may be incomplete; the next run retries it
                continue
            states[state["id"]] = state
            kind = "tiled" if state["tiled"] else "single image"
            reporter.info(
                f"[{done}/{len(todo)}] {image.source}: {state['width']}x{state['height']}, {kind}, {state['files']} files"
            )
            for note in state["notes"]:
                reporter.warn(f"{image.source}: {note}")

    result = {
        "ok": not failures,
        "destination": dest.describe(),
        "base_url": base_url,
        "processed": len(todo) - len(failures),
        "skipped": total - len(todo),
        "failed": failures,
    }
    if failures and not any(image.id in states for item in items for image in item.images):
        # Every image failed. Nothing can be published, and that is a failed run, not a usage error.
        result.update({"items": 0, "images": 0, "not_tiled": [], "orphans": []})
        return result
    counts = write_outputs(dest, base_url, items, states, config, reporter)
    catalog = load_catalog(dest)
    result.update(counts)
    result.update(summary_links(base_url, catalog))
    if args.space:
        result["space_url"] = publish_space(args.space, base_url, catalog, args.space_private)
    return result


def command_manifest(args, reporter: Reporter) -> dict:
    dest_location = parse_location(args.dest, "destination")
    dest = open_dest(dest_location, create=False, allow_private=args.allow_private)
    previous = load_catalog(dest)
    if previous is None and not args.source:
        raise UserError(
            f"Nothing is published at {args.dest} yet.",
            f"Run: iiif.py publish <SOURCE> {args.dest}",
        )
    stored_base = (previous or {}).get("config", {}).get("base_url")
    if dest_location.kind == "local" and not args.base_url and stored_base:
        base_url = stored_base
    else:
        base_url = default_base_url(dest_location, args.base_url)
    config = make_config(args, previous)
    if args.source:
        source = open_source(parse_location(args.source, "source"))
        items, _ = build_plan(source, config["title"], reporter)
    else:
        items = items_from_catalog(previous)
        old_title = previous["config"]["title"]
        for item in items:
            # Images in the source root form an item that is named after the Collection.
            if item.folder == "" and item.label == old_title:
                item.label = config["title"]
    states = load_states(dest)
    counts = write_outputs(dest, base_url, items, states, config, reporter)
    catalog = load_catalog(dest)
    result = {"ok": True, "destination": dest.describe(), "base_url": base_url}
    result.update(counts)
    result.update(summary_links(base_url, catalog))
    if args.space:
        result["space_url"] = publish_space(args.space, base_url, catalog, args.space_private)
    return result


class LocalReader:
    """Reads published files from a local destination folder."""

    def __init__(self, root: str):
        self.root = Path(root)

    def get(self, path: str) -> bytes | None:
        target = self.root / path
        return target.read_bytes() if target.is_file() else None

    def exists(self, path: str) -> bool:
        return (self.root / path).is_file()


class HttpReader:
    """Reads published files the way a viewer does: over HTTP, without a token."""

    def __init__(self, base_url: str):
        self.base_url = base_url

    def _open(self, path: str, method: str):
        url = f"{self.base_url}/{urllib.parse.quote(path)}"
        request = urllib.request.Request(url, method=method, headers={"User-Agent": "iiif-tiles-check"})
        return urllib.request.urlopen(request, timeout=30)

    def get(self, path: str) -> bytes | None:
        try:
            with self._open(path, "GET") as response:
                return response.read()
        except urllib.error.HTTPError:
            return None

    def exists(self, path: str) -> bool:
        try:
            with self._open(path, "GET") as response:
                response.read(1)
                return True
        except urllib.error.HTTPError:
            return False


def spread_sample(values: list, count: int) -> list:
    """Up to `count` values, spread evenly over the list."""
    if count <= 0 or len(values) <= count:
        return values
    step = len(values) / count
    return [values[int(number * step)] for number in range(count)]


def command_check(args, reporter: Reporter) -> dict:
    """Verify that the published files agree with each other and can be fetched."""
    location = parse_location(args.dest, "destination")
    problems = []
    if location.kind == "local":
        reader = LocalReader(location.path)
        catalog_bytes = reader.get(CATALOG_PATH)
        if catalog_bytes is None:
            raise UserError(f"No published collection in {args.dest}.", "Run publish first.")
        base_url = json.loads(catalog_bytes)["config"]["base_url"]
    else:
        base_url = default_base_url(location, None)
        reader = HttpReader(base_url)
    is_remote = location.kind == "bucket"

    def relative(url: str, what: str) -> str | None:
        if is_remote and re.search(r"//(localhost|127\.0\.0\.1)", url):
            problems.append(f"{what} is a localhost URL: {url}")
            return None
        if not url.startswith(base_url + "/"):
            problems.append(f"{what} does not start with the base URL {base_url}: {url}")
            return None
        return url[len(base_url) + 1 :]

    def load(path: str, what: str) -> dict | None:
        data = reader.get(path)
        if data is None:
            problems.append(f"{what} cannot be fetched: {path}")
            return None
        try:
            return json.loads(data)
        except ValueError:
            problems.append(f"{what} is not valid JSON: {path}")
            return None

    canvases = []
    collection = load("collection.json", "Collection")
    manifests_checked = 0
    if collection is not None:
        if collection.get("id") != f"{base_url}/collection.json":
            problems.append(f"Collection id is {collection.get('id')}, expected {base_url}/collection.json")
        for reference in collection.get("items", []):
            path = relative(reference.get("id", ""), "Manifest id in the Collection")
            if path is None:
                continue
            manifest = load(path, "Manifest")
            if manifest is None:
                continue
            manifests_checked += 1
            if manifest.get("id") != reference.get("id"):
                problems.append(f"Manifest at {path} has id {manifest.get('id')}")
            canvases.extend(manifest.get("items", []))
    reporter.info(f"Checked {manifests_checked} manifest(s) with {len(canvases)} canvas(es)")

    index_by_canvas = {}
    index_rows = None
    index_bytes = reader.get("index.parquet")
    if index_bytes is None:
        problems.append("index.parquet cannot be fetched")
    else:
        import pyarrow.parquet as pq

        columns = ["canvas_id", "image_url", "thumbnail_url", "derivative_url"]
        rows = pq.read_table(io.BytesIO(index_bytes), columns=columns).to_pylist()
        index_rows = len(rows)
        index_by_canvas = {row["canvas_id"]: row for row in rows}
        manifest_ids = {canvas.get("id") for canvas in canvases}
        if set(index_by_canvas) != manifest_ids or index_rows != len(canvases):
            only_index = len(set(index_by_canvas) - manifest_ids)
            only_manifest = len(manifest_ids - set(index_by_canvas))
            problems.append(
                f"index.parquet and the manifests disagree: {only_index} canvas id(s) only in the index, "
                f"{only_manifest} only in the manifests"
            )

    sample = spread_sample(canvases, args.sample)
    for canvas in sample:
        name = canvas.get("id", "?")
        row = index_by_canvas.get(name)
        if not canvas.get("thumbnail"):
            problems.append(f"{name}: canvas has no thumbnail")
        else:
            path = relative(canvas["thumbnail"][0]["id"], f"{name}: thumbnail")
            if path is not None and not reader.exists(path):
                problems.append(f"{name}: thumbnail cannot be fetched: {path}")
        try:
            body = canvas["items"][0]["items"][0]["body"]
            service_id = body["service"][0]["id"]
        except (KeyError, IndexError):
            problems.append(f"{name}: canvas has no image with an image service")
            continue
        path = relative(body["id"], f"{name}: image")
        if path is not None and not reader.exists(path):
            problems.append(f"{name}: full image cannot be fetched: {path}")
        if row is not None:
            if row["image_url"] != body["id"]:
                problems.append(f"{name}: index image_url is {row['image_url']}, the manifest says {body['id']}")
            if canvas.get("thumbnail") and row["thumbnail_url"] != canvas["thumbnail"][0]["id"]:
                problems.append(f"{name}: index thumbnail_url differs from the canvas thumbnail")
            path = relative(row["derivative_url"], f"{name}: index derivative_url")
            if path is not None and not reader.exists(path):
                problems.append(f"{name}: derivative image cannot be fetched: {path}")
        service_path = relative(service_id, f"{name}: image service")
        if service_path is None:
            continue
        info = load(f"{service_path}/info.json", f"{name}: info.json")
        if info is None:
            continue
        if info.get("id") != service_id:
            problems.append(f"{name}: info.json id is {info.get('id')}, the manifest says {service_id}")
        if (info.get("width"), info.get("height")) != (canvas.get("width"), canvas.get("height")):
            problems.append(f"{name}: info.json size differs from the canvas size")
        for size in info.get("sizes", []):
            size_path = f"{service_path}/full/{size['width']},{size['height']}/0/default.jpg"
            is_full = (size["width"], size["height"]) == (info.get("width"), info.get("height"))
            if not reader.exists(size_path) and not is_full:
                problems.append(f"{name}: declared size cannot be fetched: {size_path}")
        for tiles in info.get("tiles", []):
            tile_w = min(tiles["width"], info["width"])
            tile_h = min(tiles.get("height", tiles["width"]), info["height"])
            tile_path = f"{service_path}/0,0,{tile_w},{tile_h}/{tile_w},{tile_h}/0/default.jpg"
            if (tile_w, tile_h) == (info["width"], info["height"]):
                tile_path = f"{service_path}/full/max/0/default.jpg"  # canonical form of the whole image
            if not reader.exists(tile_path):
                problems.append(f"{name}: first tile cannot be fetched: {tile_path}")
            if (tile_w, tile_h) != (info["width"], info["height"]):
                # The last full-resolution tile: the bottom right corner of the image.
                size = tiles["width"]
                left = ((info["width"] - 1) // size) * size
                top = ((info["height"] - 1) // size) * size
                last_w, last_h = info["width"] - left, info["height"] - top
                tile_path = f"{service_path}/{left},{top},{last_w},{last_h}/{last_w},{last_h}/0/default.jpg"
                if not reader.exists(tile_path):
                    problems.append(f"{name}: last tile cannot be fetched: {tile_path}")
    reporter.info(f"Checked files for {len(sample)} of {len(canvases)} canvas(es)")

    for problem in problems:
        reporter.warn(problem)
    return {
        "ok": not problems,
        "base_url": base_url,
        "manifests": manifests_checked,
        "canvases": len(canvases),
        "canvases_checked": len(sample),
        "index_rows": index_rows,
        "problems": problems,
        "hint": "" if not problems else "Fix with: iiif.py manifest <DEST>   (rewrites every id and the index)",
    }


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------


def add_description_options(parser) -> None:
    parser.add_argument("--title", help="Label of the Collection (default: IIIF Collection)")
    parser.add_argument("--rights", help="Default rights URI (Creative Commons or RightsStatements.org)")
    parser.add_argument("--attribution", help="Default credit line, shown by viewers")
    parser.add_argument("--language", help="Language code of labels and metadata (default: none)")
    parser.add_argument("--base-url", help="URL that serves the destination. Required for a local destination")
    parser.add_argument("--space", help="Create or update a static Space with a viewer, for example org/name")
    parser.add_argument("--space-private", action="store_true", help="Create the viewer Space as private")
    parser.add_argument("--allow-private", action="store_true", help="Allow a private destination bucket")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    publish = commands.add_parser("publish", help="Tile images and write manifests, Collection and index")
    publish.add_argument("source", help="Local folder or hf://buckets/<owner>/<name>[/<prefix>]")
    publish.add_argument("dest", help="hf://buckets/<owner>/<name>[/<prefix>] or a local folder")
    add_description_options(publish)
    publish.add_argument("--tile-size", type=int, default=512, help="Tile size in pixels (default: 512)")
    publish.add_argument(
        "--tile-threshold",
        type=int,
        default=2048,
        help="Images with a longer edge than this get tiles (default: 2048)",
    )
    publish.add_argument("--quality", type=int, default=85, help="JPEG quality (default: 85)")
    publish.add_argument("--workers", type=int, default=4, help="Images processed at the same time (default: 4)")
    publish.add_argument("--force", action="store_true", help="Process every image again")

    manifest = commands.add_parser("manifest", help="Rebuild manifests, Collection and index without tiling")
    manifest.add_argument("dest", help="Destination of an earlier publish")
    manifest.add_argument("--source", help="Read labels and metadata from this source again")
    add_description_options(manifest)

    check = commands.add_parser("check", help="Verify a published destination")
    check.add_argument("dest", help="Destination of an earlier publish")
    check.add_argument("--sample", type=int, default=25, help="Canvases to check in detail; 0 = all (default: 25)")

    for command in (publish, manifest, check):
        command.add_argument("--json", action="store_true", help="Print the result as JSON on stdout")
    return parser


def print_result(command: str, result: dict, reporter: Reporter) -> None:
    if reporter.json_mode:
        print(json.dumps(result, indent=2))
        return
    if command == "check":
        print("OK: no problems found" if result["ok"] else f"{len(result['problems'])} problem(s) found")
        if result["hint"]:
            print(result["hint"])
        return
    print(f"\nPublished {result['images']} image(s) in {result['items']} item(s) to {result['destination']}")
    if result.get("failed"):
        print(f"FAILED: {len(result['failed'])} image(s). Fix them and run the same command again.")
    if result["not_tiled"]:
        print(f"Not published (no image files yet): {len(result['not_tiled'])} image(s). Run publish.")
    if result["orphans"]:
        print(f"Still in the destination but no longer in the source: {len(result['orphans'])} image(s).")
    if "entry_url" not in result:
        return
    print(f"IIIF {result['entry_kind']}: {result['entry_url']}")
    print(f"Index (Parquet): {result['index_url']}")
    if result.get("space_url"):
        print(f"Viewer Space:    {result['space_url']}")
    print(f"Verify with:     iiif.py check {result['destination']}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reporter = Reporter(args.json)
    commands = {"publish": command_publish, "manifest": command_manifest, "check": command_check}
    try:
        result = commands[args.command](args, reporter)
    except UserError as error:
        if args.json:
            print(json.dumps({"ok": False, "error": error.message, "hint": error.hint}, indent=2))
        else:
            print(f"error: {error.message}", file=sys.stderr)
            if error.hint:
                print(f"  next: {error.hint}", file=sys.stderr)
        return 2
    except Exception as error:
        if not args.json:
            raise
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}", "hint": ""}, indent=2))
        return 1
    print_result(args.command, result, reporter)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
