"""Stage immutable releases or verify hosted assets; never publishes anything.

Network and checksums use only the standard library. --inference additionally
requires validation-requirements.txt. Run with Python -I from a trusted directory.
"""

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
RELEASE_BASE = "https://github.com/manafishrov/detection-models/releases/download/"
SIDECARS = {
    "catalog.json": "catalog.json",
    "release-inputs.json": "release-inputs.json",
    "ultralytics-evidence.json": "license-evidence/ultralytics.json",
    "license-evidence.json": "license-evidence/marine-detect.json",
    "ULTRALYTICS-LICENSE.txt": "LICENSES/ultralytics.txt",
    "MARINE-DETECT-LICENSE.txt": "LICENSES/marine-detect.txt",
    "TOOLS-LICENSE.txt": "LICENSE",
    "NOTICE.md": "NOTICE.md",
    "SOURCE-ARCHIVES.md": "SOURCE-ARCHIVES.md",
    "release-notes.md": "release-notes.md",
}


def load_validator():
    spec = importlib.util.spec_from_file_location("validate_catalog", ROOT / "scripts/validate_catalog.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = load_validator()


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def safe_name(name: str) -> bool:
    return isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is not None


def release_data(catalog_path: Path, inputs_path: Path):
    models = json.loads(catalog_path.read_text(encoding="utf-8"))
    validator.validate(models)
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    tag = inputs["release"]
    validator.require(re.fullmatch(r"v\d+\.\d+\.\d+", tag) is not None, "invalid release tag")
    base = RELEASE_BASE + tag + "/"
    names = set(SIDECARS) | {"SHA256SUMS", "tooling-source.tar.gz"}
    for model in models:
        asset = model["asset"]
        validator.require(asset["fileName"] not in names, "duplicate release asset")
        names.add(asset["fileName"])
        validator.require(asset["url"] == base + asset["fileName"], "models must pin the same Manafish release")
        validator.require(validator.is_https(model.get("upstreamAssetUrl")), "missing original asset URL")
    for item in inputs["artifacts"]:
        name = item["fileName"]
        validator.require(safe_name(name) and name not in names, "unsafe or duplicate source artifact")
        names.add(name)
        validator.require(validator.is_sha256(item["sha256"]), f"{name}: invalid source hash")
        validator.require(type(item["fileSize"]) is int and 0 < item["fileSize"] <= validator.MAX_MODEL_BYTES,
                          f"{name}: invalid source size")
        validator.require(validator.is_https(item["url"]), f"{name}: invalid source URL")
    by_name = {item["fileName"]: item for item in inputs["artifacts"]}
    for model in models:
        weights = model["sourceWeights"]
        validator.require(weights["fileName"] in by_name, "original checkpoint missing from release")
        validator.require(by_name[weights["fileName"]]["sha256"] == weights["sha256"], "checkpoint hash mismatch")
    return models, inputs, base


def check_file(path: Path, item: dict) -> None:
    if path.stat().st_size != item["fileSize"] or sha256(path) != item["sha256"]:
        raise ValueError(f"{path.name}: size or SHA-256 mismatch")


def download(item: dict, cache: Path) -> Path:
    # Each untrusted download has its own directory; no archives are extracted.
    name = item["fileName"]
    if not safe_name(name) or not validator.is_https(item["url"]):
        raise ValueError("unsafe download filename or URL")
    directory = cache / name
    path = directory / name
    if path.exists():
        check_file(path, item)
        return path
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / (name + ".tmp")
    size = 0
    created = False
    try:
        request = Request(item["url"], headers={"User-Agent": "manafish-detection-models/1.1"})
        with urlopen(request, timeout=120) as response:
            if not validator.is_https(response.url):
                raise ValueError("download redirected away from HTTPS")
            with temporary.open("xb") as output:
                created = True
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > item["fileSize"]:
                        raise ValueError(f"{name}: download exceeds expected size")
                    output.write(chunk)
        check_file(temporary, item)
        temporary.rename(path)
    finally:
        if created:
            temporary.unlink(missing_ok=True)
    return path


def find_input(item: dict, directories: list[Path], cache: Path) -> Path:
    for directory in directories:
        for path in sorted(directory.rglob(item["fileName"])):
            if path.is_file():
                check_file(path, item)
                return path
    return download(item, cache)


def source_archive(destination: Path) -> None:
    result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    with destination.open("xb") as output, gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name in sorted(set(result.stdout.rstrip("\0").split("\0"))):
                path = ROOT / name
                if not path.exists():
                    continue
                if not path.is_file() or path.is_symlink():
                    raise ValueError(f"unsupported source file: {name}")
                content = path.read_bytes()
                info = tarfile.TarInfo("detection-models/" + name)
                info.size = len(content)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(content))


def infer(path: Path, model: dict) -> dict:
    import ast
    import numpy as np
    import onnx
    import onnxruntime as ort

    check_file(path, model["asset"])
    graph = onnx.load(str(path), load_external_data=False)
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):
        raise ValueError(f"{path.name}: external tensor data is not supported")
    onnx.checker.check_model(graph)
    metadata = {item.key: item.value for item in graph.metadata_props}
    names = ast.literal_eval(metadata["names"])
    count = len(model.get("classes", [])) or model["classCount"]
    if not isinstance(names, dict) or set(names) != set(range(count)):
        raise ValueError(f"{path.name}: incompatible class metadata")
    if "classes" in model and [names[index] for index in range(count)] != model["classes"]:
        raise ValueError(f"{path.name}: class names differ from catalog")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError(f"{path.name}: expected one input and one output")
    entry = session.get_inputs()[0]
    shape = [1, 3, 640, 640]
    if entry.type != "tensor(float)" or len(entry.shape) != 4 or any(
        isinstance(actual, int) and actual != expected for actual, expected in zip(entry.shape, shape)
    ):
        raise ValueError(f"{path.name}: incompatible input {entry}")
    output = session.run(None, {entry.name: np.full(shape, 114 / 255, dtype=np.float32)})[0]
    expected = (1, 300, 6) if model["id"].startswith("yolo26") else (1, count + 4, 8400)
    if output.shape != expected or output.dtype != np.float32 or not np.isfinite(output).all():
        raise ValueError(f"{path.name}: incompatible output {output.shape}")
    return {"id": model["id"], "sha256": model["asset"]["sha256"], "outputShape": list(output.shape),
            "onnx": onnx.__version__, "onnxruntime": ort.__version__, "numpy": np.__version__}


def checksums(directory: Path) -> str:
    return "".join(f"{sha256(path)}  {path.name}\n" for path in sorted(directory.iterdir())
                   if path.name != "SHA256SUMS")


def parse_checksums(text: str) -> dict:
    result = {}
    for line in text.splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or not validator.is_sha256(digest) or not safe_name(name) or name in result:
            raise ValueError("unsafe, duplicate or malformed SHA256SUMS entry")
        result[name] = digest
    return result


def sidecar_path(name: str, relative: str, args) -> Path:
    if name == "catalog.json":
        return args.catalog
    if name == "release-inputs.json":
        return args.inputs
    return ROOT / relative


def stage(args, models: list, inputs: dict) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for item in [model["asset"] | {"url": model["upstreamAssetUrl"]} for model in models] + inputs["artifacts"]:
        source = find_input(item, args.input_dir, args.cache_dir)
        shutil.copyfile(source, args.output_dir / item["fileName"])
    for name, relative in SIDECARS.items():
        shutil.copyfile(sidecar_path(name, relative, args), args.output_dir / name)
    source_archive(args.output_dir / "tooling-source.tar.gz")
    for model in models:
        path = args.output_dir / model["asset"]["fileName"]
        check_file(path, model["asset"])
        if args.inference:
            print(json.dumps(infer(path, model)), flush=True)
    (args.output_dir / "SHA256SUMS").write_text(checksums(args.output_dir), encoding="utf-8")
    print(f"Staged {args.output_dir}. No files were published.")


def small_download(url: str, limit: int) -> bytes:
    with urlopen(Request(url, headers={"User-Agent": "manafish-detection-models/1.1"}), timeout=120) as response:
        if not validator.is_https(response.url):
            raise ValueError("download redirected away from HTTPS")
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("release metadata exceeds size limit")
    return data


def verify(args, models: list, inputs: dict, base: str) -> None:
    tag = inputs["release"]
    metadata = json.loads(small_download(
        f"https://api.github.com/repos/manafishrov/detection-models/releases/tags/{tag}", 1024 * 1024))
    if metadata.get("tag_name") != tag or metadata.get("draft") is not False or metadata.get("immutable") is not True:
        raise ValueError("release must be published and immutable before acceptance")
    # Never trust a downloaded catalog to redefine the expected model bytes.
    published = small_download(base + "catalog.json", 1024 * 1024)
    if published != args.catalog.read_bytes():
        raise ValueError("hosted catalog differs from the reviewed local catalog")
    sums_bytes = small_download(base + "SHA256SUMS", 64 * 1024)
    sums = parse_checksums(sums_bytes.decode("utf-8"))
    pinned = {item["fileName"]: item for item in inputs["artifacts"] + [m["asset"] for m in models]}
    expected_names = set(pinned) | set(SIDECARS) | {"tooling-source.tar.gz"}
    if set(sums) != expected_names:
        raise ValueError("hosted checksum manifest has missing or unexpected assets")
    assets = metadata["assets"]
    hosted = {item["name"]: item for item in assets}
    if len(hosted) != len(assets) or set(hosted) != expected_names | {"SHA256SUMS"}:
        raise ValueError("published asset list has missing or unexpected files")
    with tempfile.TemporaryDirectory() as directory:
        archive = Path(directory) / "tooling-source.tar.gz"
        source_archive(archive)
        if sums[archive.name] != sha256(archive):
            raise ValueError("hosted tooling source differs from the reviewed repository files")
    for name, item in pinned.items():
        if sums[name] != item["sha256"]:
            raise ValueError(f"{name}: hosted checksum differs from reviewed pin")
    for name, relative in SIDECARS.items():
        if sums[name] != sha256(sidecar_path(name, relative, args)):
            raise ValueError(f"{name}: hosted file differs from the tagged source")
    sums_record = hosted["SHA256SUMS"]
    if sums_record["size"] != len(sums_bytes) or sums_record.get("digest") != "sha256:" + hashlib.sha256(sums_bytes).hexdigest():
        raise ValueError("SHA256SUMS differs from the GitHub asset digest or size")
    paths = {}
    for name, digest in sums.items():
        size = hosted[name]["size"]
        if type(size) is not int or not 0 < size <= validator.MAX_MODEL_BYTES:
            raise ValueError(f"{name}: invalid hosted asset size")
        if hosted[name].get("digest") != "sha256:" + digest:
            raise ValueError(f"{name}: GitHub asset digest differs from checksum manifest")
        if name in pinned and size != pinned[name]["fileSize"]:
            raise ValueError(f"{name}: hosted size differs from reviewed pin")
        item = {"fileName": name, "fileSize": size, "sha256": digest, "url": base + name}
        paths[name] = download(item, args.cache_dir)
        print(f"Verified {name}: {digest}", flush=True)
    if args.inference:
        for model in models:
            print(json.dumps(infer(paths[model["asset"]["fileName"]], model)), flush=True)
    print("Hosted release verified against the reviewed catalog and source pins.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "stage", "verify"])
    parser.add_argument("--catalog", type=Path, default=ROOT / "catalog.json")
    parser.add_argument("--inputs", type=Path, default=ROOT / "release-inputs.json")
    parser.add_argument("--release", help="must match the reviewed release tag")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "artifacts/downloads")
    parser.add_argument("--input-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--inference", action="store_true")
    args = parser.parse_args()
    try:
        models, inputs, base = release_data(args.catalog, args.inputs)
        if args.release is not None and args.release != inputs["release"]:
            raise ValueError("requested release does not match reviewed source")
        if args.command == "stage":
            if args.output_dir is None:
                raise ValueError("stage requires --output-dir (a new directory)")
            stage(args, models, inputs)
        elif args.command == "verify":
            verify(args, models, inputs, base)
        else:
            print("Release inputs validated; no network or ML dependencies used.")
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Release validation failed: {error}\n")


if __name__ == "__main__":
    main()
