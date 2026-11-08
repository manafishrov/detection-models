# General detectors

The catalog offers YOLO26n (Nano), YOLO26s (Small), YOLO26m (Medium), YOLO11n
(Nano), and YOLO11s (Small). All five are pretrained on the same 80 COCO object
categories. They do not include a fish category.

Nano is the lightweight starting point. Small and Medium improve benchmark
accuracy at the cost of more computation. Actual frame rate depends on the
computer, execution provider, input size, and video pipeline.

The app loads these official FP32 ONNX assets at 640×640. YOLO11n uses
dynamic axes; the other four have fixed input dimensions. The app resolves
the output dimensions through a test inference.
All five files were checked against their pinned hashes and passed the app's
CPU inference validation on 2026-11-01. This was a compatibility check, not a
new measurement of detection accuracy or GPU performance.

- [YOLO26 models and published benchmarks](https://docs.ultralytics.com/models/yolo26/)
- [YOLO11 models and published benchmarks](https://docs.ultralytics.com/models/yolo11/)
- [COCO classes](https://docs.ultralytics.com/datasets/detect/coco/)
- [Upstream license declaration](https://www.ultralytics.com/license)

The exact file sizes, source URLs, and hashes are in [the catalog](../catalog.json).
Manafish `v1.1.0` hosts the same tested ONNX bytes; `upstreamAssetUrl` retains
the official original download. [Source materials](../SOURCE-ARCHIVES.md) and
[metadata evidence](../license-evidence/ultralytics.json) distinguish the asset
release tag, ONNX exporter versions and companion checkpoint versions.
