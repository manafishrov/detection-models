import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_assets", ROOT / "scripts/release_assets.py")
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class Response(io.BytesIO):
    def __init__(self, data, url):
        super().__init__(data)
        self.url = url


class ReleaseTests(unittest.TestCase):
    def test_current_release_inputs_pass(self):
        models, inputs, base = release.release_data(ROOT / "catalog.json", ROOT / "release-inputs.json")
        self.assertEqual(len(models), 7)
        self.assertEqual(base, release.RELEASE_BASE + "v1.1.0/")
        self.assertEqual(sum(item["role"] == "original-checkpoint" for item in inputs["artifacts"]), 7)

    def test_original_general_metadata_matches_catalog_and_source_versions(self):
        evidence = json.loads((ROOT / "license-evidence/ultralytics.json").read_text())
        models, inputs, _ = release.release_data(ROOT / "catalog.json", ROOT / "release-inputs.json")
        sources = {item["version"] for item in inputs["artifacts"] if "version" in item}
        catalog = {model["id"]: model for model in models}
        self.assertEqual(len(evidence["models"]), 5)
        for item in evidence["models"]:
            model = catalog[item["modelId"]]
            self.assertEqual(item["onnx"]["sha256"], model["asset"]["sha256"])
            self.assertEqual(item["checkpoint"]["sha256"], model["sourceWeights"]["sha256"])
            self.assertEqual(item["onnx"]["upstreamAssetUrl"], model["upstreamAssetUrl"])
            for record in (item["onnx"], item["checkpoint"]):
                self.assertTrue(record["metadata"]["license"].startswith("AGPL-3.0"))
                self.assertIn(record["metadata"]["version"], sources)

    def test_release_rejects_mixed_urls_missing_provenance_and_source_drift(self):
        original_models = json.loads((ROOT / "catalog.json").read_text())
        original_inputs = json.loads((ROOT / "release-inputs.json").read_text())
        mutations = [
            lambda models, inputs: models[0]["asset"].update(url=models[0]["upstreamAssetUrl"]),
            lambda models, inputs: models[0].pop("upstreamAssetUrl"),
            lambda models, inputs: inputs["artifacts"][0].update(fileName="../unsafe.pt"),
            lambda models, inputs: inputs["artifacts"].append(inputs["artifacts"][0]),
            lambda models, inputs: inputs["artifacts"][0].update(fileSize=True),
            lambda models, inputs: models[0]["sourceWeights"].update(sha256="0" * 64),
        ]
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.json"
            manifest = Path(directory) / "release-inputs.json"
            for mutate in mutations:
                models, inputs = copy.deepcopy(original_models), copy.deepcopy(original_inputs)
                mutate(models, inputs)
                catalog.write_text(json.dumps(models))
                manifest.write_text(json.dumps(inputs))
                with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                    release.release_data(catalog, manifest)

    def test_checksum_manifest_rejects_unsafe_or_ambiguous_entries(self):
        digest = "a" * 64
        self.assertEqual(release.parse_checksums(f"{digest}  model.onnx\n"), {"model.onnx": digest})
        for text in (f"{digest}  ../model.onnx", f"{digest}  /model.onnx", f"{digest}  a/b",
                     f"{digest}  model.onnx\n{digest}  model.onnx", "bad  model.onnx",
                     f"{digest} model.onnx", f"{digest}  model.onnx\n\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                release.parse_checksums(text)

    def test_download_checks_size_hash_redirect_and_cache(self):
        content = b"model bytes"
        item = {"fileName": "test.onnx", "fileSize": len(content),
                "sha256": hashlib.sha256(content).hexdigest(), "url": "https://example.com/test.onnx"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (body, url) in enumerate(((b"wrong", item["url"]), (b"x" * len(content), item["url"]),
                                                (content + b"!", item["url"]),
                                                (content, "http://example.com/test.onnx"))):
                with patch.object(release, "urlopen", return_value=Response(body, url)):
                    with self.subTest(index=index), self.assertRaises(ValueError):
                        release.download(item, root / str(index))
                self.assertFalse(list((root / str(index)).rglob("*.tmp")))
            with patch.object(release, "urlopen", return_value=Response(content, item["url"])) as network:
                path = release.download(item, root / "valid")
                self.assertEqual(path.read_bytes(), content)
                self.assertEqual(release.download(item, root / "valid"), path)
                self.assertEqual(network.call_count, 1)
                path.write_bytes(b"corrupted cache")
                with self.assertRaises(ValueError):
                    release.download(item, root / "valid")
                self.assertEqual(network.call_count, 1)

    def test_source_archive_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("source")
            listing = subprocess.CompletedProcess([], 0, stdout="deleted.txt\0a.txt\0a.txt\0")
            with patch.object(release, "ROOT", root), patch.object(release.subprocess, "run", return_value=listing):
                release.source_archive(root / "first.tar.gz")
                release.source_archive(root / "second.tar.gz")
            self.assertEqual((root / "first.tar.gz").read_bytes(), (root / "second.tar.gz").read_bytes())
            with tarfile.open(root / "first.tar.gz") as archive:
                self.assertEqual(archive.getnames(), ["detection-models/a.txt"])
                member = archive.getmembers()[0]
                self.assertEqual((member.mtime, member.mode, member.uid, member.gid), (0, 0o644, 0, 0))

    def test_hosted_verification_uses_reviewed_pins_and_checks_every_asset(self):
        base = "https://example.com/release/"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.json"
            catalog.write_bytes(b"reviewed catalog")
            bodies = {"catalog.json": catalog.read_bytes(), "test.onnx": b"onnx bytes",
                      "test.pt": b"checkpoint bytes", "tooling-source.tar.gz": b"source archive"}
            sums = {name: hashlib.sha256(body).hexdigest() for name, body in bodies.items()}
            manifest = "".join(f"{digest}  {name}\n" for name, digest in sums.items())
            def item(name):
                return {"fileName": name, "fileSize": len(bodies[name]), "sha256": sums[name]}
            models = [{"asset": item("test.onnx")}]
            inputs = {"release": "v1.1.0", "artifacts": [item("test.pt")]}
            bodies["SHA256SUMS"] = manifest.encode()
            metadata = {"tag_name": "v1.1.0", "draft": False, "immutable": True, "assets": [
                {"name": name, "size": len(body), "digest": "sha256:" + hashlib.sha256(body).hexdigest()}
                for name, body in bodies.items()
            ]}
            def response(request, timeout):
                body = json.dumps(metadata).encode() if request.full_url.startswith("https://api.github.com/") \
                    else bodies[request.full_url.removeprefix(base)]
                return Response(body, request.full_url)
            def archive(path):
                path.write_bytes(bodies["tooling-source.tar.gz"])
            args = SimpleNamespace(catalog=catalog, cache_dir=root / "cache", inference=False)
            with patch.object(release, "ROOT", root), patch.object(release, "SIDECARS", {"catalog.json": "catalog.json"}), \
                    patch.object(release, "urlopen", side_effect=response), \
                    patch.object(release, "source_archive", side_effect=archive):
                release.verify(args, models, inputs, base)
                self.assertEqual(len(list(args.cache_dir.glob("*/*"))), 4)
                metadata["assets"].append({"name": "unexpected.txt"})
                with self.assertRaisesRegex(ValueError, "published asset list"):
                    release.verify(args, models, inputs, base)
                metadata["assets"].pop()
                original_digest = metadata["assets"][0]["digest"]
                metadata["assets"][0]["digest"] = "sha256:" + "0" * 64
                with self.assertRaisesRegex(ValueError, "GitHub asset digest"):
                    release.verify(args, models, inputs, base)
                metadata["assets"][0]["digest"] = original_digest
                original_archive = bodies["tooling-source.tar.gz"]
                bodies["tooling-source.tar.gz"] = b"different tooling source"
                with self.assertRaisesRegex(ValueError, "tooling source differs"):
                    release.verify(args, models, inputs, base)
                bodies["tooling-source.tar.gz"] = original_archive
                metadata["immutable"] = False
                with self.assertRaisesRegex(ValueError, "published and immutable"):
                    release.verify(args, models, inputs, base)
                metadata["immutable"] = True
                bodies["catalog.json"] = b"changed catalog"
                with self.assertRaisesRegex(ValueError, "catalog differs"):
                    release.verify(args, models, inputs, base)
                bodies["catalog.json"] = catalog.read_bytes()
                bodies["SHA256SUMS"] = manifest.replace(sums["test.onnx"], "0" * 64).encode()
                with self.assertRaisesRegex(ValueError, "differs from reviewed pin"):
                    release.verify(args, models, inputs, base)
                bodies["SHA256SUMS"] = (manifest + "0" * 64 + "  unexpected.txt\n").encode()
                with self.assertRaisesRegex(ValueError, "missing or unexpected"):
                    release.verify(args, models, inputs, base)


if __name__ == "__main__":
    unittest.main()
