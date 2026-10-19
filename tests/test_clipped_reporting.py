"""Check reported tail errors and raw diagnostics without GPU dependencies."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "benchmark_runner", Path(__file__).resolve().parents[1] / "benchmarks" / "run.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class FakeTorch:
    complex128 = object()

    @staticmethod
    def full(*args, **kwargs):
        return None

    class no_grad:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False


class FakeModel:
    def eval(self):
        return None

    def __call__(self, theta):
        raise AssertionError("Patched inversion should not call the transform")


class ClippedReportingTests(unittest.TestCase):
    def test_out_of_range_prediction_is_retained_but_reported_clipped(self):
        fake_mp = types.SimpleNamespace(invertlaplace=lambda *args, **kwargs: 1.2)
        with patch.dict(sys.modules, {"mpmath": fake_mp}):
            points, summaries = runner.evaluate(FakeModel(), "hcn", 20, "cpu", FakeTorch())
        self.assertEqual(len(points), 20)
        self.assertTrue(all(row["raw_prediction"] == 1.2 for row in points))
        self.assertTrue(all(row["clipped_prediction"] == 1.0 for row in points))
        self.assertTrue(all(row["raw_invalid"] for row in points))
        self.assertTrue(all(row["relative_error_percent"] > row["clipped_relative_error_percent"] for row in points))
        self.assertTrue(all(row["raw_invalid_count"] == int(row["interval"].split(":")[1]) for row in summaries))
        self.assertTrue(all(row["mre_percent"] == row["clipped_mre_percent"] for row in summaries))


if __name__ == "__main__":
    unittest.main()
