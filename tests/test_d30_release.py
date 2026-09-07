#!/usr/bin/env python3
"""Dependency-light integrity checks for the audited 30D release result."""

import csv
import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "d30_skew_symmetric_product_form"
EXPECTED_CHECKPOINT_SHA256 = (
    "0bfcd0dc11779fabd477fcf5ee9bf07772e555a9b10776618dc4e47a1106be1c"
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
    require("checkpoint_step50000.pt" in manifest, "checkpoint missing from manifest")
    for relative, expected in manifest.items():
        path = RESULT / relative
        require(path.is_file(), f"manifest file is missing: {relative}")
        require(sha256_file(path) == expected, f"manifest hash mismatch: {relative}")

    checkpoint = RESULT / "checkpoint_step50000.pt"
    require(sha256_file(checkpoint) == EXPECTED_CHECKPOINT_SHA256, "checkpoint hash mismatch")
    checkpoint_record = json.loads((RESULT / "checkpoint_sha256.json").read_text())
    require(checkpoint_record["step"] == 50_000, "checkpoint step changed")
    require(checkpoint_record["seed"] == 12345, "checkpoint seed changed")
    require(checkpoint_record["sha256"] == EXPECTED_CHECKPOINT_SHA256, "checkpoint record mismatch")

    config = json.loads((RESULT / "run_config.json").read_text())
    frozen = config["config"]
    expected = {
        "seed": 12345,
        "d": 30,
        "start_step": 20_000,
        "final_step": 50_000,
        "batch_bar": 1_024,
        "batch_bdary": 128,
        "batch_mono": 128,
        "lr_transition_after_step": 40_000,
    }
    for key, value in expected.items():
        require(frozen[key] == value, f"frozen config mismatch: {key}")
    require(config["experiment_variant"] == "d30_general_pointmatched_20d", "variant changed")
    require(config["tail_or_ground_truth_used_for_selection"] is False, "selection policy changed")
    serialized_config = json.dumps(config)
    require(not re.search(r"[A-Za-z]:\\\\", serialized_config), "local Windows path leaked")

    pointwise = read_csv("checkpoint_pointwise.csv")
    require(len(pointwise) == 30, "pointwise grid must contain 30 rows")
    grid = {(int(row["degree"]), int(row["t"])) for row in pointwise}
    require(grid == {(degree, t) for degree in (5, 6, 8) for t in range(1, 11)}, "pointwise grid mismatch")
    require("clipped_prediction" not in pointwise[0], "clipped predictions must not be released")

    history = read_csv("training_history.csv")
    require(len(history) == 90, "historical training log length changed")
    require(int(history[-1]["step"]) == 50_000, "historical endpoint changed")
    require(history[-1]["diagnostic_status"] == "ok", "historical endpoint diagnostic failed")
    for row in pointwise:
        degree, t = int(row["degree"]), int(row["t"])
        history_value = float(history[-1][f"D{degree}_tail_t{t}_evaluation_only"])
        require(
            math.isclose(history_value, float(row["raw_prediction"]), rel_tol=0.0, abs_tol=1e-15),
            f"history/pointwise mismatch at D{degree}, t={t}",
        )

    summary = read_csv("checkpoint_summary.csv")
    require({int(row["degree"]) for row in summary} == {5, 6, 8}, "summary degrees changed")
    degree8 = next(row for row in summary if int(row["degree"]) == 8)
    require(math.isclose(float(degree8["mre_1_8"]), 0.06547215830053443, rel_tol=0.0, abs_tol=1e-15), "D8 MRE(1:8) changed")
    require(math.isclose(float(degree8["mre_1_10"]), 0.0887265081510682, rel_tol=0.0, abs_tol=1e-15), "D8 MRE(1:10) changed")
    require(int(degree8["invalid_count_1_10"]) == 1, "D8 overshoot flag changed")

    exact = read_csv("exact_transform_summary.csv")
    exact8 = next(row for row in exact if int(row["degree"]) == 8)
    require(math.isclose(float(exact8["mre_1_8"]), 0.0017938378895622352, rel_tol=0.0, abs_tol=1e-15), "D8 exact floor changed")
    require(math.isclose(float(exact8["mre_1_10"]), 0.00376052651113763, rel_tol=0.0, abs_tol=1e-15), "D8 exact floor(1:10) changed")

    smoke = json.loads((RESULT / "smoke_test_report.json").read_text())
    require(smoke["passed"] is True, "public smoke test did not pass")
    require(smoke["formal_results_written"] is False, "smoke test wrote formal results")
    require(smoke["preflight"]["parameter_real_scalar_dof"] == 17_028, "parameter count changed")
    print("30D release integrity checks passed.")


if __name__ == "__main__":
    main()
