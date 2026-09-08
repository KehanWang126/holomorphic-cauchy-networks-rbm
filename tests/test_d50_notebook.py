#!/usr/bin/env python3
"""Static release checks for the public 50D training notebook."""

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "d50_skew_symmetric_product_form.ipynb"


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
    markdown = "\n\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    full_text = code + "\n\n" + markdown

    for index, cell_code in enumerate(code_cells):
        ast.parse(cell_code, filename=f"{NOTEBOOK.name}:code-cell-{index}")

    require(
        all(
            cell.get("execution_count") is None
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ),
        "committed notebook must have no execution counts",
    )
    require(
        all(
            not cell.get("outputs")
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ),
        "committed notebook must have no saved outputs",
    )

    require('RUN_MODE = "smoke"' in code, "safe smoke default is missing")
    require(
        'ALLOWED_RUN_MODES = {"smoke", "train", "evaluate"}' in code,
        "run-mode gate changed",
    )
    require("DIMENSION = 50" in code, "dimension changed")
    require("PAPER_SEED = 12345" in code, "paper seed changed")
    require("TRAIN_STEPS = 100_000" in code, "fixed training budget changed")
    require("LR_SWITCH_AFTER_STEP = 80_000" in code, "learning-rate switch changed")
    require("CHECKPOINT_STEP = 100_000" in code, "fixed endpoint changed")
    require('"batch_bar": 512' in code, "BAR batch changed")
    require('"batch_bdary": 64' in code, "boundary batch changed")
    require('"batch_mono": 64' in code, "monotonicity batch changed")
    require('"eval_batch_bar": 1_024' in code, "evaluation BAR batch changed")
    require('"eval_batch_bdary": 128' in code, "evaluation boundary batch changed")
    require('"eval_batch_mono": 128' in code, "evaluation monotonicity batch changed")
    require(
        "assert PER_RUN_SAMPLE_BUDGET == 64_000_000" in code,
        "sample-budget assertion missing",
    )
    require("18_308" in code, "parameter-count assertion missing")
    require(
        "run_sequential_backward_training_step" in code,
        "memory-bounded backward path missing",
    )
    require("RECOVERY_EVERY = 1_000" in code, "recovery schedule changed")
    require(
        '"checkpoint_selection": "prespecified fixed step 100000 only"' in code,
        "endpoint policy changed",
    )
    require('"talbot_degrees": [5, 8]' in code, "Talbot diagnostic degrees changed")
    require('"primary_degree": 8' in code, "primary Talbot degree changed")
    require('"prediction_clipping": False' in code, "raw-prediction policy changed")
    require(
        'HISTORICAL_CHECKPOINT_SCHEMA = "d50_general_pointmatched100k_v1"' in code,
        "historical fixed-checkpoint schema is missing",
    )
    require(
        'HISTORICAL_EXPERIMENT_VARIANT = "d50_controlled_general_scaling_pure_general"' in code,
        "historical training variant is missing",
    )
    require(
        "669c9bc1b2188241e47e78ee5a23bfaa690a86233529b8eef340daf11cf79b0a" in code,
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
    require(
        'cauchy_product_form_d50_general_pointmatched_step{CHECKPOINT_STEP}.pt' in code,
        "historical endpoint filename is missing",
    )

    training_position = code.index("def train_fixed_endpoint")
    truth_position = code.index("def exact_tail_probability")
    require(training_position < truth_position, "ground truth appears before training")
    require("PUBLISHED_NN4MGF" not in full_text, "unsupported 50D baseline was introduced")
    require(
        "return -mp.expm1(DIMENSION * mp.log1p(-mp.exp(-t)))" in code,
        "stable exact-tail formula is missing",
    )

    forbidden = [
        r"C:\\Users\\",
        r"/Users/",
        r"/home/",
        "RUN_TRAINING = True",
        "RUN_PREFLIGHT = True",
        "D50_DIAGNOSTIC_STEPS",
        "50k_to_86k",
        "50K_TO_86K",
        "best_valid",
        "Retrospective best",
        "checkpoint_step50000",
        "checkpoint_step86000",
    ]
    for token in forbidden:
        require(token not in full_text, f"forbidden historical/local token remains: {token}")

    require(
        not re.search(r"[A-Za-z]:\\", full_text),
        "Windows absolute path remains in notebook",
    )
    print("50D notebook release checks passed.")


if __name__ == "__main__":
    main()
