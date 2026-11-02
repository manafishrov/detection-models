# Third-party sources and license status

## Ultralytics general detectors

YOLO26 and YOLO11 are provided by Ultralytics. This catalog refers to the
official ONNX files from the `ultralytics/assets` release `v8.4.0` and records
their hashes. It does not mirror those files.

- [YOLO26 documentation](https://docs.ultralytics.com/models/yolo26/)
- [YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/)
- [Ultralytics licensing](https://www.ultralytics.com/license)
- [Official assets](https://github.com/ultralytics/assets/releases/tag/v8.4.0)

The catalog records the upstream AGPL-3.0 license declaration. Users remain
responsible for complying with the applicable model terms.

## Marine Detect

FishInv and MegaFauna were developed by Orange and Tēnaka for the Let's Revive
project. Their source documentation focuses on Malaysia/Indo-Pacific marine
environments. These are upstream models converted by Manafish, not models
trained by Manafish.

The reviewed upstream code revision is
`587f45184ab0f04ed5a243c2625c441dfb2996a3`:

- [Source repository and model download references](https://github.com/Orange-OpenSource/marine-detect/tree/587f45184ab0f04ed5a243c2625c441dfb2996a3)
- [Upstream code license](https://github.com/Orange-OpenSource/marine-detect/blob/587f45184ab0f04ed5a243c2625c441dfb2996a3/LICENSE)
- [Copy of that license](LICENSES/marine-detect.txt)

Both original checkpoint files contain the explicit metadata declaration
`AGPL-3.0 (https://ultralytics.com/license)`. The checkpoints were obtained
through the upstream README's download links and verified against the recorded
SHA-256 hashes before inspecting their metadata. This evidence comes from the
original files, not from an exporter's default license label. See
[the recorded declarations](license-evidence/marine-detect.json).

The model release retains the original checkpoints, converted ONNX models,
source manifests, license text and upstream source archives. The conversion
code and pinned dependency versions are in the release's tagged source.
The original checkpoints were produced with Ultralytics 8.2.64; the ONNX
conversion uses Ultralytics 8.4.197. Both source versions accompany the release.

The ONNX conversions are dated 2026-11-02. They preserve the checkpoint's
AGPL-3.0 declaration, add source attribution and the reviewed upstream revision,
and remove the export timestamp for reproducible model bytes. Compatibility
checks do not establish detection accuracy on Manafish ROV footage.

The upstream README credits Tēnaka for its images. This repository does not
distribute upstream images or training datasets.
