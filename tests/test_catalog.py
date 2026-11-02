import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = load_script("validate_catalog")
exporter = load_script("export_models")
CATALOG = json.loads((ROOT / "catalog.json").read_text(encoding="utf-8"))


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.catalog = copy.deepcopy(CATALOG)

    def test_current_catalog_passes(self):
        validator.validate(self.catalog)

    def test_duplicate_or_missing_entries_are_rejected(self):
        for catalog in (self.catalog[:-1], self.catalog + [self.catalog[0]]):
            with self.subTest(count=len(catalog)), self.assertRaises(ValueError):
                validator.validate(catalog)

    def test_invalid_pins_are_rejected(self):
        for field, value in (("sha256", "not-a-hash"), ("fileSize", 0),
                             ("fileSize", True), ("fileSize", validator.MAX_MODEL_BYTES + 1)):
            with self.subTest(field=field, value=value):
                catalog = copy.deepcopy(CATALOG)
                catalog[0]["asset"][field] = value
                with self.assertRaises(ValueError):
                    validator.validate(catalog)

    def test_unsafe_download_urls_and_filenames_are_rejected(self):
        for field, value in (("url", "http://example.com/model.onnx"),
                             ("url", "file:///etc/passwd"),
                             ("url", "https://user:password@example.com/model.onnx"),
                             ("fileName", "../model.onnx")):
            with self.subTest(field=field, value=value):
                catalog = copy.deepcopy(CATALOG)
                catalog[0]["asset"][field] = value
                with self.assertRaises(ValueError):
                    validator.validate(catalog)

    def test_unconfirmed_weights_cannot_be_made_downloadable(self):
        self.catalog[-1].update(license="NOASSERTION", licenseStatus="unconfirmed", licenseEvidence=None)
        self.catalog[-1]["asset"]["url"] = "https://example.com/MegaFauna.onnx"
        with self.assertRaisesRegex(ValueError, "unconfirmed weights cannot be downloadable"):
            validator.validate(self.catalog)

    def test_unconfirmed_weights_cannot_assert_a_license(self):
        self.catalog[-1].update(licenseStatus="unconfirmed", licenseEvidence=None)
        self.catalog[-1]["asset"]["url"] = None
        self.catalog[-1]["license"] = "AGPL-3.0"
        with self.assertRaisesRegex(ValueError, "unconfirmed weights cannot declare a license"):
            validator.validate(self.catalog)

    def test_declared_license_needs_evidence(self):
        self.catalog[0]["licenseEvidence"] = None
        with self.assertRaisesRegex(ValueError, "missing license evidence"):
            validator.validate(self.catalog)

    def test_marine_license_evidence_identifies_the_original_checkpoint(self):
        evidence = json.loads((ROOT / "license-evidence/marine-detect.json").read_text())
        by_id = {item["modelId"]: item for item in evidence["checkpoints"]}
        for model in self.catalog:
            if not model["experimental"]:
                continue
            declaration = by_id[model["id"]]
            self.assertEqual(declaration["sha256"], model["sourceWeights"]["sha256"])
            self.assertEqual(declaration["field"], "license")
            self.assertEqual(declaration["value"], "AGPL-3.0 (https://ultralytics.com/license)")

    def test_export_rejects_changed_checkpoint_before_importing_ml_libraries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "FishInv.pt").write_bytes(b"a changed checkpoint")
            with self.assertRaisesRegex(ValueError, "source weights failed their SHA-256 check"):
                exporter.prepare("FishInv", root, root)


if __name__ == "__main__":
    unittest.main()
