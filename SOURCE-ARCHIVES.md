# Release source materials

`release-inputs.json` pins every checkpoint and third-party source archive by
size, SHA-256 and upstream URL. `SHA256SUMS` covers all uploaded files, including
the catalog, licenses, notices, evidence, and `tooling-source.tar.gz` (this
repository's tooling, requirements, and documentation). No training datasets
are included. Ultralytics source archives are retained intact, including their
two packaged sample images; Marine Detect demonstration images are omitted.

## General detectors

The five ONNX files are byte-for-byte copies of Ultralytics' `v8.4.0` asset
release. That asset tag is **not** the exporter version. Original protobuf
metadata and checkpoint declarations are preserved in
`ultralytics-evidence.json`; the inspection script never unpickles a checkpoint.

| Models | ONNX exporter | PyTorch producer | Companion checkpoint version |
| --- | --- | --- | --- |
| YOLO26n | 8.4.38 | 2.10.0 | 8.3.222 |
| YOLO26s, YOLO26m | 8.4.41 | 2.11.0 | 8.3.222 |
| YOLO11n | 8.3.237 | 2.9.1 | 8.2.100 |
| YOLO11s | 8.4.41 | 2.11.0 | 8.2.100 |

The release includes the official same-named `.pt` companions, the Ultralytics
source archives for all five recorded versions, and the upstream AGPL license.
Each source archive is pinned to the commit behind the corresponding official
tag; full commit IDs and archive hashes are in `release-inputs.json`. Source
archives retain upstream copyright notices and licenses.

These are upstream version declarations, not proof that the public tag or the
same-named checkpoint reproduces the ONNX file. In particular, the recorded
YOLO26 checkpoint version predates public YOLO26 support. Manafish has not
reconstructed the original training environment, training data, or exact export
command beyond the metadata retained in the ONNX file. This release preserves
the tested upstream bytes rather than claiming a reproducible upstream export
or making a blanket legal assessment of source completeness.

## Marine Detect

The original `FishInv.pt`, `MegaFauna.pt`, converted ONNX files and
`prepared-models.json` are unchanged from Manafish `v1.0.0`. Their original
checkpoint declarations are in `license-evidence.json`.

- `marine-detect-source-587f451.tar.gz`: Orange-OpenSource/marine-detect at
  `587f45184ab0f04ed5a243c2625c441dfb2996a3`. Demonstration `assets/` images are
  omitted; source, requirements, documentation and license are retained.
- `ultralytics-source-8.2.64.tar.gz`: checkpoint framework at
  `b9f987ccce9f7db0759c2221b09286818dcb4cad`.
- `ultralytics-source-8.4.197.tar.gz`: Manafish exporter at
  `1d85e0ddedd7754fdbf238cc941b57b170a36b67`.

These three archives are also retained unchanged from `v1.0.0`. The conversion
script and pinned requirements are in `tooling-source.tar.gz`. Export options
are FP32, batch 1, static 640×640, opset 17, without simplification; the script
preserves source attribution and removes the timestamp. `NOTICE.md` records
the conversion date, upstream credit and limitations. The checkpoint license
declaration and the marine-detect code license are distinct evidence.
