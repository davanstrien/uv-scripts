---
viewer: false
tags:
  - uv-script
  - iiif
  - glam
  - deep-zoom
  - static-tiles
---

# Publish images as a static IIIF collection

Turn a folder or a bucket of images into a [IIIF](https://iiif.io/) collection that a Hugging Face Storage Bucket serves as plain files. No image server.

One command gives you:

- deep-zoom images ([IIIF Image API 3.0](https://iiif.io/api/image/3.0/), level 0),
- one manifest per item and a Collection ([Presentation API 3.0](https://iiif.io/api/presentation/3.0/)), with your labels, metadata and rights,
- `index.parquet`, one row per image, for programs,
- optionally a Space with a viewer to share.

## Quick start

```bash
uv run https://huggingface.co/datasets/uv-scripts/iiif-tiles/raw/main/iiif.py \
    publish ./my-scans hf://buckets/myorg/my-collection \
    --title "My collection" --space myorg/my-collection
```

The command prints the IIIF entry URL (the Collection, or the manifest when there is one item), the index URL and the Space link. The script creates the bucket if it does not exist. Viewers read the bucket without a token, so it must be public; the script stops on a private bucket unless you pass `--allow-private`.

Run it again after you add images or change metadata. Images that did not change are skipped.

## Run on Hugging Face Jobs

For a large collection, or when the images are already in a bucket:

```bash
hf jobs uv run --flavor cpu-upgrade --timeout 2h --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/iiif-tiles/raw/main/iiif.py \
    publish hf://buckets/myorg/source-images hf://buckets/myorg/my-collection \
    --title "My collection"
```

`--secrets HF_TOKEN` forwards your token so the Job can write to the bucket. If a Job stops, run the same command again: it skips every image that was finished and continues with the rest.

Images in other cloud storage: copy them into a Hugging Face bucket first (`hf buckets sync ./folder hf://buckets/myorg/source-images`), then publish from that bucket.

## How to lay out the source

```
my-scans/
├── items.csv            optional: one row per folder
├── metadata.csv         optional: one row per image
├── letter-1/            one folder = one item = one manifest
│   ├── page1.jpg
│   └── page2.jpg
├── letter-2/
│   └── page1.tif
└── loose-photo.jpg      images in the root form one more item
```

- Pages are ordered by name, and `page2` comes before `page10`.
- Supported formats: JPEG, TIFF, PNG, WebP.
- This is the same layout as a Hub [ImageFolder](https://huggingface.co/docs/datasets/image_dataset#imagefolder) dataset, so `metadata.csv` (or `metadata.jsonl`) uses a `file_name` column.

`metadata.csv`, one row per image:

```csv
file_name,label,date,rights
letter-1/page1.jpg,Page 1 recto,1936,http://creativecommons.org/publicdomain/mark/1.0/
```

`items.csv`, one row per folder:

```csv
folder,label,author,attribution
letter-1,Letter to a friend,A. Writer,My Library
```

| Column | Use |
|---|---|
| `label` | Name shown in viewers. Default: the file or folder name |
| `rights` | A [Creative Commons](https://creativecommons.org/) or [RightsStatements.org](https://rightsstatements.org/) URI. It is stored in the `http://` form that IIIF uses |
| `attribution` | Credit line that viewers must show |
| any other column | Shown as metadata. Image columns are also added to the index as `meta_<column>` |

`--rights` and `--attribution` on the command line set the default for the whole collection.

## Use the collection from a program

Viewers use the manifests. Programs should use the index: it is one file, and it has a direct URL for every image.

```python
import duckdb

index = "https://huggingface.co/buckets/myorg/my-collection/resolve/index.parquet"
duckdb.sql(f"SELECT label, derivative_url FROM '{index}' WHERE meta_date = '1936'")
```

| Column | Content |
|---|---|
| `image_id`, `item_id`, `position`, `label`, `item_label` | Identity and order |
| `width`, `height`, `tiled` | Size in pixels, after rotation, and whether the image has tiles |
| `canvas_id` | IIIF canvas id. Use it with `#xywh=x,y,w,h` to point at a region |
| `manifest_url`, `image_service` | IIIF entry points |
| `image_url` | Full-size JPEG |
| `derivative_url`, `derivative_width`, `derivative_height` | JPEG with a long edge of 1024 px: a good input size for most models. Images are never enlarged, so a smaller image is given at full size |
| `thumbnail_url` | JPEG with a long edge of 400 px, or the full image if it is smaller |
| `source_path`, `rights`, `attribution`, `meta_*` | From your source and metadata files |

For bulk work, sync the bucket (`hf buckets sync hf://buckets/myorg/my-collection ./local`) instead of fetching files one by one. Static files cannot crop a region on request: download `derivative_url` or `image_url` and crop locally.

The fixed columns above are never removed or renamed. The `meta_*` columns follow your metadata files.

## Change labels or metadata later

```bash
# Read the metadata files again and rewrite manifests, Collection and index. No image is processed.
uv run iiif.py manifest hf://buckets/myorg/my-collection --source ./my-scans

# Change the title only
uv run iiif.py manifest hf://buckets/myorg/my-collection --title "A better title"
```

## Verify

```bash
uv run iiif.py check hf://buckets/myorg/my-collection
```

`check` reads the collection the way a viewer does. It reads every manifest and compares the canvas ids with the index. For a sample of 25 images (`--sample 0` for all) it also confirms that each id matches its URL and that the full image, thumbnail, 1024 px image and the first and last tile can be fetched. The exit code is 1 when it finds a problem. It makes 16 requests at a time (`--workers`) and tries a failed request again; 164 images with `--sample 0` took about 20 seconds.

## For scripts and agents

- `--json` prints one JSON result on stdout. Progress goes to stderr.
- The script never asks a question.
- Exit codes: 0 = success, 1 = some images failed, `check` found problems, or an unexpected error; 2 = the command cannot run as given. Most errors name the next step.
- URLs are predictable: they are built from the destination, the image id and the image size.

## Options

`publish SOURCE DEST`

| Flag | Default | Description |
|---|---|---|
| `--title` | IIIF Collection | Label of the Collection |
| `--rights`, `--attribution` | none | Defaults for the whole collection |
| `--language` | `none` | Language code of your labels and metadata, for example `en` |
| `--space` | none | Create or update a static Space with a viewer (`--space-private` to keep it private) |
| `--tile-size` | 512 | Tile size in pixels |
| `--tile-threshold` | 2048 | Images with a longer edge than this get tiles. Smaller images get no tiles |
| `--quality` | 85 | JPEG quality |
| `--workers` | 4 | Images processed at the same time |
| `--force` | off | Process every image again |
| `--base-url` | bucket URL | URL that serves the destination. Required for a local destination |

`SOURCE` and `DEST` are a local folder or `hf://buckets/<owner>/<name>[/<prefix>]`.

## What the destination contains

```
collection.json                  the Collection: start here
manifests/<item>.json            one manifest per item
images/<image-id>/info.json      IIIF image description
images/<image-id>/full/...       full size, 1024 px and 400 px JPEGs
images/<image-id>/<region>/...   tiles, for large images
index.parquet                    one row per image
_publish/                        this tool's records; viewers do not read them
```

An image id is made from the source path: `letter-1/page1.jpg` becomes `letter-1__page1`. Each folder and file name becomes lowercase ASCII letters, digits and `-`, and `__` joins them. Two files that would get the same id stop the run with an error.

**These URLs are permanent once you share them.** They contain the bucket name and the image id. If you rename the bucket, a folder or a file, the URLs usually change. After a bucket rename, run `manifest` once to rewrite the ids.

## Things to know

- **Rotation.** A photo with an EXIF rotation tag is stored upright. Width and height are the upright values.
- **Colour.** Every file is saved as 8-bit sRGB (or greyscale for a greyscale image). An embedded colour profile such as Adobe RGB is applied in the conversion. CMYK and 16-bit images are converted, and transparency is flattened onto white; the run prints a warning for each of these.
- **Multi-page TIFF.** Only the first page is published. The run prints a warning. A pyramidal TIFF is read as one image.
- **The destination belongs to the tool.** Do not keep your own files under `images/`, `manifests/` or `_publish/`: a run can delete them.
- **Source and destination must not overlap.** The destination cannot be inside the source, or the reverse.
- **Removed images.** An image that you delete from the source leaves the manifests on the next run. Its image files stay in the bucket, and the run reports the count.
- **Failed images.** An image that cannot be read is left out, the run exits with code 1, and the other images are published. Fix the file and run the same command again.
- **Python.** The script needs Python 3.10 to 3.13 (the image library has no build for newer versions yet). `uv` selects one for you.
- **Viewer.** The Space uses [Triiiceratops](https://github.com/d-flood/triiiceratops), pinned to one version. Other IIIF viewers can open the Collection or manifest URL.
- **Level 0.** All files are made in advance. That gives deep zoom and browsing. It does not give crops, rotation or other sizes on request; those need an image server.

## Local test

```bash
uv run iiif.py publish ./my-scans ./out --base-url http://localhost:8000
cd out && python -m http.server 8000
```

Browser viewers on another address also need CORS headers, which `http.server` does not send. A bucket sends them.

## Older script: `tile_iiif.py`

`tile_iiif.py` is the first version of this recipe. It writes tiles and one flat manifest, without metadata, thumbnails or an index. It stays here so existing commands keep working. Use `iiif.py` for new work.
