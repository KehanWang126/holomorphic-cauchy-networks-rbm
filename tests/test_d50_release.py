#!/usr/bin/env python3
"""Dependency-light integrity checks for the audited 50D release result."""

import csv
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "d50_skew_symmetric_product_form"
EXPECTED_CHECKPOINT_SHA256 = (
    "669c9bc1b2188241e47e78ee5a23bfaa690a86233529b8eef340daf11cf79b0a"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(name):
    with (RESULT / name).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    manifest = {}
    for line in (RESULT / "manifest.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        manifest[relative] = digest
    require("checkpoint_step100000.pt" in manifest, "checkpoint missing from manifest")
    for relative, expected in manifest.items():
        path = RESULT / relative
        require(path.is_file(), f"manifest file is missing: {relative}")
        require(sha256_file(path) == expected, f"manifest hash mismatch: {relative}")

    checkpoint = RESULT / "checkpoint_step100000.pt"
    require(sha256_file(checkpoint) == EXPECTED_CHECKPOINT_SHA256, "checkpoint hash mismatch")
    checkpoint_record = json.loads((RESULT / "checkpoint_sha256.json").read_text())
    require(checkpoint_record["step"] == 100_000, "checkpoint step changed")
    require(checkpoint_record["seed"] == 12345, "checkpoint seed changed")
    require(checkpoint_record["sha256"] == EXPECTED_CHECKPOINT_SHA256, "checkpoint record mismatch")

    # Inspect only pickle bytes and archive names; never execute the historical pickle.
    with zipfile.ZipFile(checkpoint) as archive:
        pickle_member = next(name for name in archive.namelist() if name.endswith("/data.pkl"))
        pickle_bytes = archive.read(pickle_member)
        storage_members = [name for name in archive.namelist() if "/data/" in name]
    require(b"d50_general_pointmatched100k_v1" in pickle_bytes, "checkpoint schema missing")
    require(
        b"d50_controlled_general_scaling_pure_general" in pickle_bytes,
        "checkpoint variant missing",
    )
    require(len(storage_members) == 77, "checkpoint tensor-storage count changed")

    config = json.loads((RESULT / "run_config.json").read_text())
    frozen = config["config"]
    expected = {
        "seed": 12345,
        "eval_seed": 1012348,
        "d": 50,
        "start_step": 0,
        "final_step": 100_000,
        "batch_bar": 512,
        "batch_bdary": 64,
        "batch_mono": 64,
        "lr_transition_after_step": 80_000,
    }
    for key, value in expected.items():
        require(frozen[key] == value, f"frozen config mismatch: {key}")
    require(
        config["experiment_variant"] == "d50_controlled_general_scaling_pure_general",
        "variant changed",
    )
    require(config["tail_or_ground_truth_used_for_selection"] is False, "selection policy changed")
    require(not re.search(r"[A-Za-z]:\\\\", json.dumps(config)), "local Windows path leaked")

    pointwise = read_csv("checkpoint_pointwise.csv")
    require(len(pointwise) == 20, "pointwise grid must contain 20 rows")
    grid = {(int(row["degree"]), int(row["t"])) for row in pointwise}
    require(grid == {(degree, t) for degree in (5, 8) for t in range(1, 11)}, "pointwise grid mismatch")
    require("clipped_prediction" not in pointwise[0], "clipped predictions must not be released")
    degree8_t2 = next(row for row in pointwise if int(row["degree"]) == 8 and int(row["t"]) == 2)
    require(degree8_t2["invalid_flag"] == "True", "D8 t=2 overshoot flag changed")
    require(math.isclose(float(degree8_t2["raw_prediction"]), 1.000477639318975, rel_tol=0.0, abs_tol=1e-15), "D8 t=2 prediction changed")

    history = read_csv("training_history.csv")
    require(len(history) == 90, "historical training log length changed")
    require(int(history[0]["step"]) == 1_000, "historical first record changed")
    require(int(history[-1]["step"]) == 100_000, "historical endpoint changed")
    require(history[-1]["diagnostic_status"] == "ok", "historical endpoint diagnostic failed")
    require(math.isclose(float(history[-1]["lr"]), 1e-5, rel_tol=0.0, abs_tol=0.0), "final learning rate changed")

    summary = read_csv("checkpoint_summary.csv")
    require({int(row["degree"]) for row in summary} == {5, 8}, "summary degrees changed")
    degree8 = next(row for row in summary if int(row["degree"]) == 8)
    require(math.isclose(float(degree8["mre_1_5"]), 0.015042789768449055, rel_tol=0.0, abs_tol=1e-15), "D8 MRE(1:5) changed")
    require(math.isclose(float(degree8["mre_1_8"]), 0.06373004212886592, rel_tol=0.0, abs_tol=1e-15), "D8 MRE(1:8) changed")
    require(math.isclose(float(degree8["mre_1_10"]), 0.08015375498068039, rel_tol=0.0, abs_tol=1e-15), "D8 MRE(1:10) changed")
    require(int(degree8["invalid_count_1_10"]) == 1, "D8 invalid count changed")

    exact = read_csv("exact_transform_summary.csv")
    exact8 = next(row for row in exact if int(row["degree"]) == 8)
    require(math.isclose(float(exact8["mre_1_8"]), 0.004687362714980669, rel_tol=0.0, abs_tol=1e-15), "D8 exact floor changed")
    require(math.isclose(float(exact8["mre_1_10"]), 0.006076124875849044, rel_tol=0.0, abs_tol=1e-15), "D8 exact floor(1:10) changed")

    parameter = json.loads((RESULT / "parameter_budget_audit.json").read_text())
    require(parameter["trainable_real_scalar_dof"] == 18_308, "parameter count changed")
    budget = json.loads((RESULT / "sample_budget.json").read_text())
    require(budget["total_sampled_constraints"] == 64_000_000, "sample budget changed")

    smoke = json.loads((RESULT / "smoke_test_report.json").read_text())
    require(smoke["passed"] is True, "public smoke test did not pass")
    require(smoke["formal_results_written"] is False, "smoke test wrote formal results")
    require(smoke["preflight"]["parameter_real_scalar_dof"] == 18_308, "smoke parameter count changed")

    provenance = json.loads((RESULT / "source_provenance.json").read_text())
    require(provenance["released_checkpoint"]["used_by_tail_diagnostic"] is True, "diagnostic endpoint identity changed")
    require(provenance["release_assembly"]["model_inference_rerun"] is False, "recomputation claim changed")

    for name in ["run_config.json", "source_provenance.json", "checkpoint_pointwise.csv"]:
        text = (RESULT / name).read_text(encoding="utf-8")
        require(not re.search(r"[A-Za-z]:\\\\", text), f"local Windows path leaked: {name}")
    print("50D release integrity checks passed.")


if __name__ == "__main__":
    main()
