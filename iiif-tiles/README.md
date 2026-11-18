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

The command prints the Collection URL, the index URL, a Mirador link and the Space link. The script creates the bucket if it does not exist. The bucket must be public, because viewers read it without a token.

Run it again after you add images or change metadata. Images that did not change are skipped.

## Run on Hugging Face Jobs

For a large collection, or when the images are already in a bucket:

```bash
hf jobs uv run --flavor cpu-upgrade --timeout 2h --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/iiif-tiles/raw/main/iiif.py \
    publish hf://buckets/myorg/source-images hf://buckets/myorg/my-collection \
    --title "My collection"
```

`--secrets HF_TOKEN` forwards your token so the Job can write to the bucket. A stopped Job loses no work: run the same command again and it continues.

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
| `rights` | A [Creative Commons](https://creativecommons.org/) or [RightsStatements.org](https://rightsstatements.org/) URI |
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
| `width`, `height` | Size in pixels, after rotation |
| `canvas_id` | IIIF canvas id. Use it with `#xywh=x,y,w,h` to point at a region |
| `manifest_url`, `image_service` | IIIF entry points |
| `image_url` | Full-size JPEG |
| `derivative_url`, `derivative_width`, `derivative_height` | JPEG with a long edge of 1024 px: a good input size for most models |
| `thumbnail_url` | JPEG with a long edge of 400 px |
| `source_path`, `rights`, `attribution`, `meta_*` | From your source and metadata files |

For bulk work, sync the bucket (`hf buckets sync hf://buckets/myorg/my-collection ./local`) instead of fetching files one by one. Static files cannot crop a region on request: download `derivative_url` or `image_url` and crop locally.

Columns are only ever added to the index, so code that reads it keeps working.

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

`check` reads the collection the way a viewer does. It confirms that every id matches its URL, that images, thumbnails and tiles can be fetched, and that the index agrees with the manifests. The exit code is 1 when it finds a problem.

## For scripts and agents

- `--json` prints one JSON result on stdout. Progress goes to stderr.
- The script never asks a question.
- Exit codes: 0 = success, 1 = some images failed or `check` found problems, 2 = the command cannot run. An error names the next step.
- Output URLs depend only on the bucket name and the source paths.

## Options

`publish SOURCE DEST`

| Flag | Default | Description |
|---|---|---|
| `--title` | IIIF Collection | Label of the Collection |
| `--rights`, `--attribution` | none | Defaults for the whole collection |
| `--language` | `none` | Language code of your labels and metadata, for example `en` |
| `--space` | none | Create or update a static Space with a viewer (`--space-private` to keep it private) |
| `--tile-size` | 512 | Tile size in pixels |
| `--tile-threshold` | 2048 | Images with a longer edge than this get tiles. Smaller images are one file |
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

An image id is made from the source path: `letter-1/page1.jpg` becomes `letter-1__page1`. Ids use lowercase ASCII letters, digits and `-`. Two files that would get the same id stop the run with an error.

**These URLs are permanent once you share them.** They contain the bucket name and the image id. If you rename the bucket, a folder or a file, the URLs change. After a bucket rename, run `manifest` once to rewrite the ids.

## Things to know

- **Rotation.** A photo with an EXIF rotation tag is stored upright. Width and height are the upright values.
- **Colour.** CMYK and 16-bit images are converted to 8-bit sRGB, and transparency is flattened onto white. The run prints a warning for each such image.
- **Multi-page TIFF.** Only the first page is published. The run prints a warning.
- **Removed images.** An image that you delete from the source leaves the manifests on the next run. Its files stay in the bucket, and the run reports the count.
- **Python.** The script needs Python 3.10 to 3.13 (the image library has no build for newer versions yet). `uv` selects one for you.
- **Viewer.** The Space uses [Triiiceratops](https://github.com/d-flood/triiiceratops), pinned to one version, and links to Mirador. Any IIIF viewer can open the Collection URL.
- **Level 0.** All files are made in advance. That gives deep zoom and browsing. It does not give crops, rotation or other sizes on request; those need an image server.

## Local test

```bash
uv run iiif.py publish ./my-scans ./out --base-url http://localhost:8000
cd out && python -m http.server 8000
```

Browser viewers on another address also need CORS headers, which `http.server` does not send. A bucket sends them.

## Older script: `tile_iiif.py`

`tile_iiif.py` is the first version of this recipe. It writes tiles and one flat manifest, without metadata, thumbnails or an index. It stays here so existing commands keep working. Use `iiif.py` for new work.
