"""Record general-model provenance without executing checkpoint pickle code.

Run with Python -I and the pinned validation-requirements.txt environment.
Downloads are inputs only; neither checkpoints nor ONNX metadata are rewritten.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickletools
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def inspect(onnx_path: Path, checkpoint_path: Path) -> dict:
    import onnx

    graph = onnx.load(str(onnx_path), load_external_data=False)
    with zipfile.ZipFile(checkpoint_path) as archive:
        members = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(members) != 1 or archive.getinfo(members[0]).file_size > 16 * 1024 * 1024:
            raise ValueError("expected one bounded checkpoint metadata pickle")
        strings = []
        for opcode, value, _ in pickletools.genops(archive.read(members[0])):
            if opcode.name in ("BINUNICODE", "SHORT_BINUNICODE", "UNICODE"):
                strings.append(value)
            if len(strings) == 8:
                break
    if strings[::2] != ["date", "version", "license", "docs"]:
        raise ValueError("unexpected checkpoint metadata prefix; inspect it manually")
    metadata = dict(zip(strings[::2], strings[1::2]))
    if not metadata["license"].startswith("AGPL-3.0"):
        raise ValueError("checkpoint does not contain the expected license declaration")
    return {
        "modelId": onnx_path.stem,
        "onnx": {
            "fileName": onnx_path.name,
            "sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
            "producer": {"name": graph.producer_name, "version": graph.producer_version},
            "opsets": {item.domain: item.version for item in graph.opset_import},
            "metadata": {item.key: item.value for item in graph.metadata_props},
        },
        "checkpoint": {
            "fileName": checkpoint_path.name,
            "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "metadata": metadata,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="contains one subdirectory per downloaded asset, named after that asset")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models = json.loads((ROOT / "catalog.json").read_text())
    evidence = {
        "method": "ONNX protobuf metadata and the leading literal date/version/license/docs strings in original checkpoint data.pkl; pickletools disassembly only, no unpickling or model execution.",
        "assetRelease": "https://github.com/ultralytics/assets/releases/tag/v8.4.0",
        "limitations": "Checkpoint files are the official same-named upstream companions. Their exact use in producing these ONNX files has not been independently established. Version strings identify upstream source tags, not a complete original build environment or a byte-reproduction guarantee.",
        "models": [],
    }
    for model in models:
        if model["experimental"]:
            continue
        name = model["asset"]["fileName"]
        checkpoint = model["sourceWeights"]["fileName"]
        item = inspect(args.input_dir / name / name, args.input_dir / checkpoint / checkpoint)
        if item["onnx"]["sha256"] != model["asset"]["sha256"]:
            raise ValueError(f"{name}: unexpected ONNX hash")
        if item["checkpoint"]["sha256"] != model["sourceWeights"]["sha256"]:
            raise ValueError(f"{checkpoint}: unexpected checkpoint hash")
        item["onnx"]["upstreamAssetUrl"] = model["upstreamAssetUrl"]
        item["checkpoint"]["upstreamAssetUrl"] = model["sourceWeights"]["url"]
        evidence["models"].append(item)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
