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

That repository includes AGPL-3.0 license text, but its README does not expressly
state whether those terms cover the externally hosted model checkpoints.
The scope of the weight license is therefore recorded as unconfirmed.
The included license text documents the upstream code license; it is not
evidence of permission to redistribute the weights.

The original checkpoints and locally prepared ONNX files are not distributed
by this repository. Their hashes and technical descriptions identify the
reviewed files without granting rights to them. Before redistribution, confirm
the applicable weight terms and identify the required Corresponding Source.
Any eventual publication must preserve applicable upstream notices, identify
the conversion and its date, and make the required source materials available.

The upstream README credits Tēnaka for its images. This repository does not
distribute upstream images or training datasets.
