# Manafish detection models

The curated object detection models used by the [Manafish ROV desktop app](https://github.com/manafishrov/app).
This repository holds the catalog, model documentation, source references, and
conversion tools. Large model files belong in versioned release assets.

## Available models

The catalog contains YOLO26n, YOLO26s, YOLO26m, YOLO11n, YOLO11s, Marine Detect
FishInv, and Marine Detect MegaFauna.

The five general detectors download directly from the official Ultralytics
`v8.4.0` release. Their 80 COCO classes include everyday objects such as people
and boats, but do not include fish. See [general detectors](models/general.md).

The two marine detectors have locally tested ONNX exports, but their downloads
remain disabled while the redistribution terms for the separately hosted
weights are unconfirmed. See [FishInv](models/marine-fishinv.md) and
[MegaFauna](models/marine-megafauna.md).

## Catalog

[`catalog.json`](catalog.json) pins model IDs, filenames, byte counts, SHA-256
hashes, source references, and license status. An asset with a `null` URL is
unavailable for download. `NOASSERTION` means the weight license has not been
established; it does not grant redistribution permission.

The app embeds a reviewed snapshot of this manifest. Updating the catalog here
does not change installed apps. Copy a reviewed catalog revision into the app,
run its native model compatibility checks, and release the app through its
usual process. The app downloads only models selected by the user and checks
their size, hash, and ONNX compatibility before installing them.

## Local checks

Python 3.11 or newer is sufficient for the catalog checks and unit tests:

```sh
python3 scripts/validate_catalog.py --check-tracked-files
python3 -m unittest discover -s tests -v
git diff --check
```

GitHub Actions runs these checks on pushes and pull requests. CI performs no
model downloads, conversions, or publishing.

## Prepare marine exports

Obtain the original `FishInv.pt` and `MegaFauna.pt` from the
[upstream model links](https://github.com/Orange-OpenSource/marine-detect/tree/587f45184ab0f04ed5a243c2625c441dfb2996a3#-model-details).
Keep the weight files outside Git. Their expected hashes are recorded in the
catalog and checked before the export tool loads a checkpoint.

Use an isolated Python 3.13 environment with
[`export-requirements.txt`](export-requirements.txt). Install its CPU PyTorch
build from the [official CPU wheel index](https://download.pytorch.org/whl/cpu).
The preparation environment uses headless OpenCV; the desktop app requires
neither Python nor these export dependencies.

With [uv](https://docs.astral.sh/uv/reference/cli/#uv-pip-install), the environment
can be prepared with:

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python --torch-backend cpu \
  --exclude export-excludes.txt -r export-requirements.txt
```

```sh
python scripts/export_models.py \
  --weights-dir /path/to/original-checkpoints \
  --output-dir artifacts/marine
```

The tool exports single-file FP32 ONNX models with 640×640 input and opset 17,
checks their graphs, runs CPU inference, and writes `prepared-models.json`.
It never uploads files. Exported weight metadata does not assert a license.

## Model releases

Marine releases require confirmation of the weight license and a review of
the Corresponding Source requirements. The upstream code license alone is
recorded separately from the unresolved weight license. See [NOTICE.md](NOTICE.md).

Once those terms are established, prepare a release containing the ONNX files,
the original checkpoints and other required Corresponding Source, the export
script and tool versions, applicable licenses and copyright notices, dated
conversion notes, and the manifest of hashes. Retain the upstream credit and
the models' geographic and accuracy limitations.

Publish a new version for changed model bytes. Verify the hosted files against
their hashes before setting their fixed download URLs in the catalog. Keep
model releases in this repository so they remain independent of app updates.

## License

Manafish's tooling and documentation in this repository use
[AGPL-3.0-or-later](LICENSE). Each third-party model retains its own terms.
The repository license does not relicense third-party weights or training data.
