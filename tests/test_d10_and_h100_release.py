"""Validate the released pilot endpoint and measured H100 profile provenance."""

import ast
import base64
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import unittest
import zlib


ROOT = Path(__file__).resolve().parents[1]


class PilotAndProfileReleaseTests(unittest.TestCase):
    def test_10d_endpoint_and_clipped_pilot(self):
        nb = json.loads((ROOT / "notebooks/d10_correlated_nonproduct_hcn.ipynb").read_text())
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                ast.parse("".join(cell["source"]))
        config_cell = "".join(nb["cells"][2]["source"])
        self.assertIn('RUN_MODE = "smoke"', config_cell)
        self.assertIn("RUN_DIR = None", config_cell)
        self.assertIn("SEED = 12345", config_cell)
        self.assertNotIn("C:\\Users", config_cell)
        folder = ROOT / "results/d10_correlated_nonproduct/seed12345"
        endpoint = folder / "checkpoint_step25000.pt"
        self.assertEqual(hashlib.sha256(endpoint.read_bytes()).hexdigest(),
                         "32603c1c46c3f6e239807c60a2395535ac9bfa90800f7cd2743ed5e2f480db26")
        raw = folder / "pointwise_raw.csv"
        self.assertEqual(raw.read_bytes(), (ROOT / "paper/audit/inputs/10D_nonproduct/hcn.csv").read_bytes())
        with raw.open(newline="") as f:
            points = [row for row in csv.DictReader(f) if row["degree"] == "8"]
        self.assertEqual(len(points), 8)
        self.assertEqual([int(row["t"]) for row in points], list(range(1, 9)))
        errors = [100 * abs(min(1.0, max(0.0, float(row["prediction"]))) - float(row["reference"]))
                  / float(row["reference"]) for row in points]
        self.assertAlmostEqual(statistics.mean(errors[:5]), 1.709661741732589)
        self.assertAlmostEqual(statistics.mean(errors), 3.68388107783899)
        self.assertEqual(sum(row["invalid"] == "True" for row in points), 1)

    def test_h100_notebook_sources_and_reports(self):
        nb = json.loads((ROOT / "benchmarks/benchmark_30d_same_h100_profile200.ipynb").read_text())
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                ast.parse("".join(cell["source"]))
        source = "".join(nb["cells"][4]["source"])
        lock = json.loads((ROOT / "benchmarks/sources.json").read_text())["sha256"]
        for name, path in (("_HCN_B64", "notebooks/d30_skew_symmetric_product_form.ipynb"),
                           ("_NN4MGF_B64", "benchmarks/vendor/nn4mgf/fit_mgf.py")):
            match = re.search(r"^" + name + r" = '([^']+)'", source, re.M)
            self.assertIsNotNone(match)
            payload = zlib.decompress(base64.b64decode(match.group(1)))
            self.assertEqual(hashlib.sha256(payload).hexdigest(), lock[path])
        folder = ROOT / "benchmarks/measurements/h100_30d_profile200"
        h = json.loads((folder / "hcn_report.json").read_text())
        n = json.loads((folder / "nn4mgf_report.json").read_text())
        self.assertEqual((h["measured_updates"], n["measured_updates"]), (200, 200))
        self.assertEqual((h["warmup_updates"], n["warmup_updates"]), (10, 10))
        self.assertEqual(h["gpu"], n["gpu"])
        self.assertAlmostEqual(n["mean_update_seconds"] / h["mean_update_seconds"], 9.696674359348206)
        self.assertAlmostEqual(n["projected_full_training_seconds_from_profile"] /
                               h["projected_full_training_seconds_from_profile"], 58.18004615608924)


if __name__ == "__main__":
    unittest.main()
