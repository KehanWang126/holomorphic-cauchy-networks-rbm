#!/usr/bin/env python3
"""Static release checks for the public 30D training notebook."""

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "d30_skew_symmetric_product_form.ipynb"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    require(notebook.get("nbformat") == 4, "notebook must use nbformat 4")
    require(len(notebook.get("cells", [])) == 15, "unexpected notebook cell count")

    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    code = "\n\n".join(code_cells)
    for index, cell_code in enumerate(code_cells):
        ast.parse(cell_code, filename=f"{NOTEBOOK.name}:code-cell-{index}")

    require(
        all(cell.get("execution_count") is None for cell in notebook["cells"] if cell.get("cell_type") == "code"),
        "committed notebook must have no execution counts",
    )
    require(
        all(not cell.get("outputs") for cell in notebook["cells"] if cell.get("cell_type") == "code"),
        "committed notebook must have no saved outputs",
    )
    require('RUN_MODE = "smoke"' in code, "safe smoke default is missing")
    require('ALLOWED_RUN_MODES = {"smoke", "train", "evaluate"}' in code, "run-mode gate changed")
    require("DIMENSION = 30" in code, "dimension changed")
    require("PAPER_SEED = 12345" in code, "paper seed changed")
    require("TRAIN_STEPS = 50_000" in code, "fixed training budget changed")
    require("LR_SWITCH_AFTER_STEP = 40_000" in code, "learning-rate switch changed")
    require("CHECKPOINT_STEP = 50_000" in code, "fixed endpoint changed")
    require('"batch_bar": 1_024' in code, "BAR batch changed")
    require('"batch_bdary": 128' in code, "boundary batch changed")
    require('"batch_mono": 128' in code, "monotonicity batch changed")
    require("assert PER_RUN_SAMPLE_BUDGET == 64_000_000" in code, "sample-budget assertion missing")
    require("17_028" in code, "parameter-count assertion missing")
    require("run_sequential_backward_training_step" in code, "memory-bounded backward path missing")
    require("RECOVERY_EVERY = 1_000" in code, "recovery schedule changed")
    require('"checkpoint_selection": "prespecified fixed step 50000 only"' in code, "endpoint policy changed")
    require('"talbot_degrees": [5, 6, 8]' in code, "Talbot diagnostic degrees changed")
    require('"primary_degree": 8' in code, "primary Talbot degree changed")
    require('"prediction_clipping": False' in code, "raw-prediction policy changed")
    require(
        'HISTORICAL_CHECKPOINT_SCHEMA = "d30_general_pointmatched50k_v1"' in code,
        "historical fixed-checkpoint schema is missing",
    )
    require(
        "0bfcd0dc11779fabd477fcf5ee9bf07772e555a9b10776618dc4e47a1106be1c" in code,
        "historical fixed-checkpoint SHA-256 changed",
    )
    require(
        "validate_historical_fixed_checkpoint" in code,
        "historical checkpoint validation path is missing",
    )
    require(
        "Refusing legacy pickle loading for an unverified checkpoint" in code,
        "hash-gated legacy loading is missing",
    )
    require(
        "new_evaluation_output_dir" in code,
        "non-overwriting evaluation output path is missing",
    )
    require("PUBLISHED_NN4MGF_DEGREE5" in code, "public baseline table missing")
    require("1956666902281f6ce4558ca82d8d15fa038f06fb" in code, "baseline provenance changed")

    training_position = code.index("def train_fixed_endpoint")
    truth_position = code.index("def exact_tail_probability")
    baseline_position = code.index("PUBLISHED_NN4MGF_DEGREE5")
    require(training_position < truth_position, "ground truth appears before training")
    require(training_position < baseline_position, "public baseline appears before training")

    forbidden = [
        r"C:\\Users\\",
        r"/Users/",
        r"/home/",
        "POINTMATCHED_SOURCE_CHECKPOINT",
        "RUN_GENERAL_60K",
        "GENERAL_60K_RESUME_FROM",
        "best_valid_D6_8",
        "Retrospective best",
    ]
    for token in forbidden:
        require(token not in code, f"forbidden historical/local token remains: {token}")

    require(
        not re.search(r"[A-Za-z]:\\", code),
        "Windows absolute path remains in notebook",
    )
    print("30D notebook release checks passed.")


if __name__ == "__main__":
    main()
