# Manafish detection models

The curated object detection models used by the [Manafish ROV desktop app](https://github.com/manafishrov/app).
This repository holds the catalog, documentation, source references and release
tools. Large files belong in versioned GitHub release assets, never Git.

## Available models

`v1.1.0` hosts all seven tested models: YOLO26n, YOLO26s, YOLO26m, YOLO11n,
YOLO11s, Marine Detect FishInv and Marine Detect MegaFauna. Every ONNX file
retains its existing bytes and SHA-256; `v1.0.0` is unchanged.

The five [general detectors](models/general.md) retain the official Ultralytics
ONNX files from the `v8.4.0` asset release. Their 80 COCO classes include people
and boats, but not fish. [FishInv](models/marine-fishinv.md) and
[MegaFauna](models/marine-megafauna.md) remain experimental. Their original
checkpoints explicitly declare AGPL-3.0. Evidence and limitations are in
[NOTICE.md](NOTICE.md) and [SOURCE-ARCHIVES.md](SOURCE-ARCHIVES.md).

## Catalog and app compatibility

[`catalog.json`](catalog.json) pins model IDs, filenames, byte counts, SHA-256
hashes, fixed Manafish release URLs, original `upstreamAssetUrl` values, source
references and license status. An asset with a `null` URL is unavailable;
`NOASSERTION` does not grant redistribution permission.

The app embeds an explicitly reviewed, tested catalog snapshot, not `main` or
`latest`. Publishing models does not update installed apps. Adopt a catalog by
pinning its release URL and catalog SHA-256 in the app, running the app's native
model compatibility tests, and releasing through the app's usual process.
Accept a model release only after its **Verify release** workflow passes. The
app still checks each user-selected download's size, hash and ONNX compatibility.

## Local checks

Python 3.11 or newer is sufficient for offline checks:

```sh
python3 -I scripts/validate_catalog.py --check-tracked-files
python3 -I scripts/release_assets.py check
python3 -I -m unittest discover -s tests -v
git diff --check
```

The lightweight **CI / ci** aggregate check follows the other Manafish
repositories. Pull requests do not download models or install ML libraries.
Renovate updates action pins and Python tooling requirements only; it does not
change catalogs, model bytes or source archive pins.

## Stage v1.1.0

Use Python 3.13 and optionally install the small pinned CPU validation stack:

```sh
python3.13 -m venv .venv
.venv/bin/python -I -m pip install -r validation-requirements.txt
```

Stage a **new** directory, reusing existing verified downloads when available:

```sh
.venv/bin/python -I scripts/release_assets.py stage \
  --input-dir artifacts/release-v1.0.0 \
  --input-dir artifacts/upstream-v1.1.0 \
  --output-dir artifacts/release-v1.1.0 \
  --inference
```

The inputs are optional caches; missing files are downloaded from the pinned
upstream URLs. The script checks every size and hash before copying, never
executes checkpoints or extracts archives, never overwrites a release directory,
and never uploads anything. It stages seven ONNX files, seven original `.pt`
checkpoints, eight third-party source archives, marine export records, original
metadata evidence, licenses, notices, catalog, tooling source and `SHA256SUMS`.
The tooling source tarball uses sorted paths and fixed timestamps/permissions
for reproducibility from identical repository files. Re-run staging after any
source/documentation changes so it describes the exact published revision.

The general models are mirrored, not re-exported. To independently regenerate
their metadata evidence from the official per-asset download directories:

```sh
.venv/bin/python -I scripts/inspect_upstream.py \
  --input-dir artifacts/upstream-v1.1.0 --output /path/to/new-evidence.json
```

Read the source limitations before publishing. Original upstream training data
and exact original export environments have not been reconstructed. Including
versioned source and metadata does not establish a blanket legal conclusion.

## Publish and accept a release

Publishing is manual and requires explicit authorization. Enable immutable
releases in GitHub settings before publication; repository settings are managed
separately, not by these workflows. Never replace assets, move published tags,
or modify `v1.0.0`. Changed model bytes, catalog corrections or missing assets
require a new model release.

After review, commit the source, merge it and tag the exact reviewed revision.
Confirm the staged source matches that revision. In this repository, the
publisher can then create a draft, upload the complete set and publish it:

```sh
gh release create v1.1.0 --repo manafishrov/detection-models \
  --verify-tag --draft --title 'Detection models v1.1.0' \
  --notes-file release-notes.md artifacts/release-v1.1.0/*
# Inspect the draft and its complete asset list before the irreversible step.
gh release edit v1.1.0 --repo manafishrov/detection-models --draft=false
```

The **Verify release** workflow runs on publication, checks out the release tag,
requires GitHub to report the release as published and immutable, and checks the
complete asset list and GitHub digests. It downloads the catalog and every
checksummed asset, compares model/source pins and a reproducible tooling archive
to the reviewed source, then runs graph checks plus finite CPU inference on all
seven ONNX files. It has read-only permissions and cannot publish, repair or
mutate a release. Re-run it manually against that tag with:

```sh
gh workflow run verify-release.yaml --repo manafishrov/detection-models \
  -f release=v1.1.0
```

The same checks can be run locally from the tagged source:

```sh
.venv/bin/python -I scripts/release_assets.py verify \
  --release v1.1.0 --cache-dir artifacts/hosted-v1.1.0 --inference
```

Use a fresh cache to independently test hosted downloads; a reused cache is
rehashed but avoids downloading unchanged files. A failure blocks app adoption;
for an immutable published release, fix forward with a new version. ONNX smoke
checks prove neither accuracy on ROV footage nor GPU/device compatibility.

## Reproduce marine exports

Obtain `FishInv.pt` and `MegaFauna.pt` through the
[upstream model links](https://github.com/Orange-OpenSource/marine-detect/tree/587f45184ab0f04ed5a243c2625c441dfb2996a3#-model-details)
or the pinned original release assets. The export tool checks their catalog
hashes before loading them. Use an isolated Python 3.13 environment with
[`export-requirements.txt`](export-requirements.txt); install CPU PyTorch from
its [official wheel index](https://download.pytorch.org/whl/cpu). With `uv`:

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python --torch-backend cpu \
  --exclude export-excludes.txt -r export-requirements.txt
.venv/bin/python -I scripts/export_models.py \
  --weights-dir /path/to/original-checkpoints --output-dir artifacts/marine
```

The exporter preserves the checkpoint license, adds source attribution, removes
the timestamp and validates FP32 640×640 opset-17 outputs. It records tool
versions in `prepared-models.json` and never publishes. The app needs none of
these Python dependencies.

## License

Manafish's tooling and documentation use [AGPL-3.0-or-later](LICENSE). Each
third-party model retains its own terms. This repository's license does not
relicense third-party weights or training data. Preserve applicable upstream
licenses, copyright notices, source materials and attribution when redistributing.
