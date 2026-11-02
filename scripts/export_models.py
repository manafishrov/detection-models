"""Prepare Marine Detect ONNX downloads locally; never uploads or publishes.

Use export-requirements.txt in an isolated Python 3.13 environment.
The desktop app does not need Python. Obtain the original weights from the
upstream references in README.md before running this tool.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


SOURCE = "https://github.com/Orange-OpenSource/marine-detect"
MODEL_IDS = {"FishInv": "marine-fishinv", "MegaFauna": "marine-megafauna"}
ROOT = Path(__file__).resolve().parents[1]
CATALOG = {model["id"]: model for model in json.loads((ROOT / "catalog.json").read_text(encoding="utf-8"))}


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def prepare(name: str, weights_dir: Path, output_dir: Path) -> dict:
    entry = CATALOG[MODEL_IDS[name]]
    weights = (weights_dir / f"{name}.pt").resolve(strict=True)
    source_digest = sha256(weights)
    if source_digest != entry["sourceWeights"]["sha256"]:
        raise ValueError(f"{name}: source weights failed their SHA-256 check")

    # Exporting must not install dependencies or download replacement weights.
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_CONFIG_DIR"] = str((output_dir / ".export-config").resolve())
    from ultralytics import YOLO
    import ultralytics
    import torch
    import onnx
    import onnxruntime as ort
    import numpy as np

    model = YOLO(str(weights), task="detect")
    checkpoint_license = model.ckpt.get("license", "")
    if entry["licenseStatus"] == "declared-upstream" and not checkpoint_license.startswith(entry["license"]):
        raise ValueError(f"{name}: checkpoint license does not match the catalog declaration")
    exported = Path(model.export(
        format="onnx", imgsz=640, batch=1, dynamic=False, simplify=False,
        opset=17, quantize=32, device="cpu",
    ))
    graph = onnx.load(str(exported))
    onnx.checker.check_model(graph)
    metadata = {item.key: item.value for item in graph.metadata_props}
    # Export timestamps otherwise change an identical model's download hash.
    metadata.pop("date", None)
    # Exporter defaults are not evidence of the upstream checkpoint's license.
    if entry["licenseStatus"] == "unconfirmed":
        metadata.pop("license", None)
    else:
        metadata["license"] = entry["license"]
    metadata.update({"author": "Orange / Tēnaka", "source": SOURCE,
                     "source_revision": entry["sourceRevision"]})
    onnx.helper.set_model_props(graph, metadata)
    destination = output_dir / f"{name}.onnx"
    staged = output_dir / f"{name}.onnx.tmp"
    onnx.save_model(graph, str(staged), save_as_external_data=False)
    session = ort.InferenceSession(str(staged), providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError(f"{name}: expected a single-input, single-output detector")
    model_input = session.get_inputs()[0]
    if model_input.type != "tensor(float)" or model_input.shape != [1, 3, 640, 640]:
        raise ValueError(f"{name}: incompatible input {model_input}")
    result = session.run(None, {model_input.name: np.full((1, 3, 640, 640), 114 / 255, dtype=np.float32)})[0]
    if result.ndim != 3 or result.shape[1] != len(model.names) + 4 or not np.isfinite(result).all():
        raise ValueError(f"{name}: incompatible detection output {result.shape}")
    del session
    staged.replace(destination)
    return {
        "id": MODEL_IDS[name], "sourceUrl": SOURCE,
        "sourceFile": weights.name, "sourceSha256": source_digest,
        "sourceRevision": entry["sourceRevision"],
        "license": entry["license"], "licenseStatus": entry["licenseStatus"],
        "checkpointLicense": checkpoint_license,
        "exportTools": {"ultralytics": ultralytics.__version__,
                        "torch": torch.__version__,
                        "onnx": onnx.__version__, "onnxruntime": ort.__version__},
        "classes": model.names, "outputShape": list(result.shape),
        "asset": {"fileName": destination.name, "fileSize": destination.stat().st_size,
                  "sha256": sha256(destination), "url": None},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", choices=list(MODEL_IDS), action="append")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "prepared-models.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name in args.model or MODEL_IDS:
        manifest[MODEL_IDS[name]] = prepare(name, args.weights_dir, args.output_dir)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {manifest_path}. No files were published.")


if __name__ == "__main__":
    main()
