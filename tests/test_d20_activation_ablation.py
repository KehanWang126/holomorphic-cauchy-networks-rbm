"""Self-contained CPU validation for the frozen 20D activation ablation."""
import ast
import copy
import csv
import json
import math
import os
from pathlib import Path
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(os.environ.get(
    "D20_TEST_PACKAGE", str(ROOT / "experiments/d20_activation_ablation")
)).resolve()
NOTEBOOK_NAME = os.environ.get("D20_TEST_NOTEBOOK", "d20_activation_ablation_evaluate.ipynb")
NB = json.loads((PACKAGE / NOTEBOOK_NAME).read_text())
os.chdir(PACKAGE)
NS = {"__name__": "test_subject"}
for cell in NB["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        if isinstance(source, list):
            source = "".join(source)
        source = source.replace('RUN_MODE = "evaluate"', 'RUN_MODE = "check"')
        exec(compile(source, "notebook_under_test", "exec"), NS)
torch = NS["torch"]
import numpy as np


def payload(entry):
    return torch.load(PACKAGE / entry["path"], map_location="cpu", weights_only=True)


def reference_rows():
    path = ROOT / "results/d20_activation_ablation/pointwise.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class EvaluationTests(unittest.TestCase):
    def test_six_preflights(self):
        self.assertEqual(len(NS["PREFLIGHT"]), 6)
        self.assertEqual([r["real_parameter_dof"] for r in NS["PREFLIGHT"]], [16388]*3+[15108]*3)

    def test_no_training_or_unsafe_load(self):
        for cell in NB["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            source = cell["source"]
            if isinstance(source, list):
                source = "".join(source)
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotIn(node.func.attr, ("backward", "step", "save", "Adam", "AdamW"))
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == "torch" and node.func.attr == "load":
                        self.assertTrue(any(k.arg == "weights_only" and isinstance(k.value, ast.Constant) and k.value.value is True for k in node.keywords))

    def test_source_definitions_unchanged(self):
        packaged = {}
        for cell in NB["cells"]:
            if cell["cell_type"] == "code":
                source = cell["source"]
                if isinstance(source, list):
                    source = "".join(source)
                for node in ast.parse(source).body:
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        packaged[node.name] = ast.dump(node, include_attributes=False)
        names = {"CauchyComplex", "CoordinateCauchyScalarNet", "run_scalar_net_chunked",
                 "StructuredCoordinateCauchyMGF", "ComplexTanh", "CoordinateTanhScalarNet",
                 "StructuredCoordinateTanhMGF"}
        cauchy_names = {
            "CauchyComplex", "CoordinateCauchyScalarNet", "run_scalar_net_chunked",
            "StructuredCoordinateCauchyMGF",
        }
        main = json.loads((ROOT / "notebooks/d20_skew_symmetric_product_form.ipynb").read_text())
        found = set()
        for cell in main["cells"]:
            if cell["cell_type"] != "code":
                continue
            try:
                tree = ast.parse("".join(cell["source"]))
            except SyntaxError:
                continue
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in cauchy_names:
                    self.assertEqual(packaged[node.name], ast.dump(node, include_attributes=False))
                    found.add(node.name)
        self.assertEqual(found, cauchy_names)
        self.assertTrue({"ComplexTanh", "CoordinateTanhScalarNet", "StructuredCoordinateTanhMGF"} <= set(packaged))

    def test_seed_mismatch_rejected(self):
        entry = NS["CHECKPOINTS"][0]
        p = payload(entry)
        p["seed"] = 999
        with self.assertRaises(ValueError):
            NS["validate_config"](p, entry)

    def test_wrong_endpoint_rejected(self):
        entry = NS["CHECKPOINTS"][0]
        p = payload(entry)
        p["last_completed_step"] = 24000
        with self.assertRaises(ValueError):
            NS["validate_config"](p, entry)

    def test_wrong_training_budget_rejected(self):
        entry = NS["CHECKPOINTS"][0]
        p = payload(entry)
        p["training_config"]["batch_bar"] = 1024
        with self.assertRaises(ValueError):
            NS["validate_config"](p, entry)

    def test_wrong_sampler_rejected(self):
        entry = NS["CHECKPOINTS"][0]
        p = payload(entry)
        p["sampler_config"]["real_domain"] = [0, 5]
        with self.assertRaises(ValueError):
            NS["validate_config"](p, entry)

    def test_wrong_method_rejected(self):
        entry = NS["CHECKPOINTS"][-1]
        p = payload(entry)
        p["method"] = "A_Cauchy"
        with self.assertRaises(ValueError):
            NS["validate_config"](p, entry)

    def test_hash_substitution_rejected(self):
        entry = dict(NS["CHECKPOINTS"][0], sha256="0"*64)
        with self.assertRaises(ValueError):
            NS["load_fixed_model"](entry)

    def test_missing_checkpoint_rejected(self):
        with self.assertRaises(FileNotFoundError):
            NS["checked_path"]("checkpoints/not_a_checkpoint.pt")

    def test_legacy_nested_seed_caveat_is_recorded(self):
        notes = [r["notes"] for r in NS["PREFLIGHT"] if r["method"] == "tanh" and r["seed"] != 12345]
        self.assertEqual(len(notes), 2)
        self.assertTrue(all(any("config.seed" in n for n in group) for group in notes))
        entry = NS["CHECKPOINTS"][-1]
        p = payload(entry)
        p["config"]["seed"] = 777
        with self.assertRaises(ValueError):
            NS["validate_config"](p, entry)

    def test_all_grid_points_present(self):
        self.assertEqual(len(ROWS), 120)
        keys = {(r["method"],int(r["seed"]),int(r["degree"]),int(r["t"])) for r in ROWS}
        self.assertEqual(keys, {(m,s,d,t) for m in ("Cauchy","tanh") for s in NS["SEEDS"] for d in (5,8) for t in range(1,11)})

    def test_pointwise_regression_against_supplied_outputs(self):
        expected = {
            (r["method"], int(r["seed"]), int(r["degree"]), int(r["t"])):
                float(r["raw_prediction"])
            for r in reference_rows()
        }
        self.assertEqual(len(expected),120)
        for r in ROWS:
            key = (r["method"],int(r["seed"]),int(r["degree"]),int(r["t"]))
            # CPU/GPU complex128 arithmetic need not be bitwise identical.
            self.assertTrue(math.isclose(float(r["raw_prediction"]),expected[key],rel_tol=1e-8,abs_tol=1e-10),(key,r["raw_prediction"],expected[key]))

    def test_hashes_unchanged(self):
        for entry in NS["CHECKPOINTS"]:
            self.assertEqual(NS["sha256_file"](PACKAGE/entry["path"]),entry["sha256"])

    def test_negative_prediction_not_clipped(self):
        group = copy.deepcopy(TYPED_ROWS[:20])
        selected = next(r for r in group if r["degree"] == 5 and r["t"] == 1)
        selected.update(raw_prediction=-1.0,relative_error=2.0,invalid_prediction=1)
        summary = NS["summarize"](group)
        row = next(r for r in summary if r["degree"]==5 and r["interval"]=="1:5")
        expect = np.mean([r["relative_error"] for r in group if r["degree"]==5 and r["t"]<=5])
        self.assertAlmostEqual(row["mre"],expect)
        self.assertGreaterEqual(row["invalid_count"],1)

    def test_nonfinite_error_not_dropped(self):
        group = copy.deepcopy(TYPED_ROWS[:20])
        group[0].update(relative_error=float("nan"),nonfinite_prediction=1,invalid_prediction=1)
        summary = NS["summarize"](group)
        row = next(r for r in summary if r["degree"] == group[0]["degree"] and r["interval"]=="1:10")
        self.assertTrue(math.isnan(row["mre"]))
        self.assertEqual(row["nonfinite_count"],1)

    def test_incomplete_grid_rejected(self):
        with self.assertRaises(ValueError):
            NS["summarize"](TYPED_ROWS[:-1])

    def test_missing_seed_rejected(self):
        summary = NS["summarize"](TYPED_ROWS)
        with self.assertRaises(ValueError):
            NS["aggregate"]([r for r in summary if r["seed"] != "32345"])

    def test_sample_standard_deviation(self):
        summary = NS["summarize"](TYPED_ROWS)
        combined = NS["aggregate"](summary)
        first = combined[0]
        values = [r["mre"] for r in summary if r["method"]==first["method"] and r["degree"]==first["degree"] and r["interval"]==first["interval"]]
        self.assertAlmostEqual(first["sample_sd_percentage_points"],100*np.std(values,ddof=1))

    def test_tanh_extreme_result_is_retained(self):
        rows = [r for r in TYPED_ROWS if r["method"]=="tanh" and r["seed"]==22345 and r["degree"]==8]
        self.assertGreater(max(r["relative_error"] for r in rows),1e8)

    def test_tanh_evaluation_does_not_call_torch_tanh(self):
        entry = next(e for e in NS["CHECKPOINTS"] if e["method"] == "tanh" and e["seed"] == 22345)
        with mock.patch.object(torch, "tanh", side_effect=AssertionError("torch.tanh must not be called")):
            model, audit = NS["load_fixed_model"](entry)
            value = model.tail_transform(NS["mp"].mpc(1, 2))
        self.assertEqual(audit["evaluation_backend"], "numpy_complex128")
        self.assertTrue(NS["mp"].isfinite(value))

    def test_archived_reference_not_claimed_recomputed(self):
        rows,provenance = NS["archived_baseline"]()
        self.assertEqual(len(rows),20)
        self.assertFalse(provenance["checkpoint_included"])
        self.assertTrue(all("NOT re-evaluated" in r["evaluation_origin"] for r in rows))


if __name__ == "__main__":
    # Recompute every endpoint directly from the six fixed weights. No training,
    # output-directory selection, or archived prediction lookup is performed.
    ROWS = []
    for entry in NS["CHECKPOINTS"]:
        model, _ = NS["load_fixed_model"](entry)
        ROWS.extend(NS["evaluate_transform"](
            entry["method"], entry["seed"], NS["learned_tail_transform"](model)
        ))
        del model
    TYPED_ROWS = copy.deepcopy(ROWS)
    for r in TYPED_ROWS:
        for k in ("seed","degree","t","invalid_prediction","nonfinite_prediction"):
            r[k] = int(r[k])
        r["relative_error"] = float(r["relative_error"])
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(EvaluationTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
