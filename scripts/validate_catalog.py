"""Validate the app-compatible model catalog without network or ML dependencies."""

import argparse
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
MAX_MODEL_BYTES = 512 * 1024 * 1024
EXPECTED_IDS = {
    "yolo26n", "yolo26s", "yolo26m", "yolo11n", "yolo11s",
    "marine-fishinv", "marine-megafauna",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def is_https(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return parsed.scheme == "https" and bool(parsed.hostname) and parsed.username is None


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate(models: object) -> None:
    require(isinstance(models, list), "catalog must be an array")
    seen = set()
    for model in models:
        require(isinstance(model, dict), "each model must be an object")
        model_id = model.get("id")
        require(isinstance(model_id, str) and model_id in EXPECTED_IDS, "unknown model ID")
        require(model_id not in seen, f"duplicate model ID: {model_id}")
        seen.add(model_id)
        for field in ("name", "license", "sourceRevision"):
            require(isinstance(model.get(field), str) and bool(model[field]), f"{model_id}: missing {field}")
        require(is_https(model.get("sourceUrl")), f"{model_id}: source URL must use HTTPS")
        require(isinstance(model.get("experimental"), bool), f"{model_id}: experimental must be boolean")
        asset = model.get("asset")
        require(isinstance(asset, dict), f"{model_id}: asset metadata is required")
        name = asset.get("fileName")
        require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_-]+\.onnx", name) is not None,
                f"{model_id}: unsafe asset filename")
        size = asset.get("fileSize")
        require(type(size) is int and 0 < size <= MAX_MODEL_BYTES, f"{model_id}: invalid asset size")
        require(is_sha256(asset.get("sha256")), f"{model_id}: invalid SHA-256")
        require("url" in asset, f"{model_id}: asset URL must be explicit")
        require(asset["url"] is None or is_https(asset["url"]), f"{model_id}: download URL must use HTTPS")
        status = model.get("licenseStatus")
        require(status in ("declared-upstream", "unconfirmed"), f"{model_id}: unknown license status")
        if status == "unconfirmed":
            require(model["license"] == "NOASSERTION", f"{model_id}: unconfirmed weights cannot declare a license")
            require(asset["url"] is None, f"{model_id}: unconfirmed weights cannot be downloadable")
            require(model.get("licenseEvidence") is None, f"{model_id}: unresolved license evidence must be null")
            weights = model.get("sourceWeights")
            require(isinstance(weights, dict) and is_sha256(weights.get("sha256")), f"{model_id}: missing source weight hash")
        else:
            require(model["license"] != "NOASSERTION", f"{model_id}: missing declared license")
            require(is_https(model.get("licenseEvidence")), f"{model_id}: missing license evidence")
    require(seen == EXPECTED_IDS, "catalog must contain all seven approved model IDs")


def check_tracked_files() -> None:
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True)
    for name in result.stdout.split("\0"):
        require(not name.lower().endswith((".pt", ".pth", ".onnx", ".onnx.data")),
                f"model binary must not be tracked in Git: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "catalog.json")
    parser.add_argument("--check-tracked-files", action="store_true")
    args = parser.parse_args()
    try:
        validate(json.loads(args.catalog.read_text(encoding="utf-8")))
        if args.check_tracked_files:
            check_tracked_files()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Catalog validation failed: {error}\n")
    print("Catalog validated: seven models; unavailable weight licenses remain gated.")


if __name__ == "__main__":
    main()
