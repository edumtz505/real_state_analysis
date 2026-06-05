"""
main.py

Command-line entry point for the analysis pipeline.
 
Usage examples:

    # Full pipeline: clean -> EDA -> hedonic -> tree models -> comparison
    # -> heterogeneity -> robustness
    python -m main all

    # One step at a time
    python -m main prepare         # clean and persist data/fotocasa_clean.csv
    python -m main eda             # descriptive analysis
    python -m main hedonic         # OLS hedonic regressions (cluster SE)
    python -m main rf              # Random Forest
    python -m main xgb             # XGBoost
    python -m main compare         # consolidated comparison
    python -m main heterogeneity   # interaction tests (type, size, region)
    python -m main robustness      # spec-sensitivity table for the agency coef

The "prepare" step has to be run at least once. After that, the other steps
read the cleaned CSV (``data/fotocasa_clean.csv``) directly.
"""
 
import argparse
import sys
import time
 
import comparison
import config as cfg
import data_loader as dl
import eda
import hedonic
import tree_models
import heterogeneity
import robustness

 
def step_prepare() -> None:
    print(f"[prepare] reading {cfg.RAW_CSV}")
    df = dl.load_raw()
    df = dl.clean(df)
    dl.summary(df)
    dl.write_clean(df)
    print(f"[prepare] cleaned data written to {cfg.CLEAN_CSV}")
 
 
def _load_or_die():
    """Load the cleaned dataset, or fall back to cleaning on the fly."""
    if cfg.CLEAN_CSV.exists():
        return dl.load_clean()
    if cfg.RAW_CSV.exists():
        print(f"[info] {cfg.CLEAN_CSV} not found; running prepare on the fly.")
        df = dl.clean(dl.load_raw())
        dl.write_clean(df)
        return df
    raise SystemExit(
        f"Neither {cfg.CLEAN_CSV} nor {cfg.RAW_CSV} exists.\n"
        f"Place the raw CSV at {cfg.RAW_CSV} and run `python -m main prepare`."
    )
 
 
def step_eda() -> None:
    eda.run_full_eda(_load_or_die())
 
 
def step_hedonic() -> None:
    hedonic.run(_load_or_die())
 
 
def step_rf() -> None:
    tree_models.run(_load_or_die(), models=("rf",))
 
 
def step_xgb() -> None:
    tree_models.run(_load_or_die(), models=("xgb",))
 
 
def step_compare() -> None:
    comparison.run()


def step_heterogeneity() -> None:
    heterogeneity.run(_load_or_die())


def step_robustness() -> None:
    robustness.run(_load_or_die())


def step_all() -> None:
    t0 = time.time()
    step_prepare()
    df = _load_or_die()
    eda.run_full_eda(df)
    hedonic.run(df)
    tree_models.run(df, models=("rf", "xgb"))
    comparison.run()
    heterogeneity.run(df)
    robustness.run(df)
    print(f"\n[all] pipeline finished in {time.time() - t0:.1f}s.")

 
_STEPS = {
    "prepare":       step_prepare,
    "eda":           step_eda,
    "hedonic":       step_hedonic,
    "rf":            step_rf,
    "xgb":           step_xgb,
    "compare":       step_compare,
    "heterogeneity": step_heterogeneity,
    "robustness":    step_robustness,
    "all":           step_all,
}
 
 
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="real-estate-pipeline",
        description="Reproducible pipeline for intermediation research.",
    )
    parser.add_argument(
        "step", choices=sorted(_STEPS),
        help="Pipeline step to run.",
    )
    args = parser.parse_args(argv)
    _STEPS[args.step]()
    return 0
 

_DEBUG_STEP = "all"


if __name__ == "__main__":
    argv = None if len(sys.argv) > 1 else [_DEBUG_STEP]
    sys.exit(main(argv))