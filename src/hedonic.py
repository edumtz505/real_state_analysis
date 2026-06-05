import warnings
 
import numpy as np
import pandas as pd
import matplotlib
 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
 
import config as cfg
import features as ft
 
warnings.filterwarnings("ignore")


_SIZE_TERMS = {
    "linear": ["Metros", "Metros2"],
    "log": ["log_Metros"],
}

def _build_formula(
    target: str,
    *,
    spec: str,
    base_numeric: list,
    dummies: list,
    treatment: str,
    fe_var: str,
    categorical_vars: list = None,
    time_fe_var: str = None,
) -> str:
    """Build a patsy formula for a given hedonic specification.
    """
    size_terms = _SIZE_TERMS[spec]
    rhs_terms = size_terms + base_numeric + dummies + [treatment]
    formula = f"{target} ~ " + " + ".join(rhs_terms)
    for cat in (categorical_vars or []):
        formula += f" + C({cat})"
    formula += f" + C({fe_var})"
    if time_fe_var:
        formula += f" + C({time_fe_var})"
    return formula
 
 
def _extract_treatment_results(result, treatment: str) -> dict:
    """Pull the rows we care about (the treatment coef) out of a fitted model."""
    ci = result.conf_int().loc[treatment]
    return {
        "coef": float(result.params[treatment]),
        "se": float(result.bse[treatment]),
        "tstat": float(result.tvalues[treatment]),
        "pval": float(result.pvalues[treatment]),
        "ci_lo": float(ci[0]),
        "ci_hi": float(ci[1]),
        "r2_adj": float(result.rsquared_adj),
        "nobs": int(result.nobs),
        "aic": float(result.aic),
    }


KEY_COEF_VARS = [
    "Inmobiliaria",
    "log_Metros",
    "Metros",
    "Metros2",
    "Habitaciones",
    "Aseos",
    "Terraza",
    "Piscina",
    "Garaje",
]

_VAR_DISPLAY = {
    "Inmobiliaria": "Inmobiliaria (treatment)",
    "log_Metros": "log(Metros)",
    "Metros": "Metros",
    "Metros2": "Metros²",
    "Habitaciones": "Habitaciones",
    "Aseos": "Aseos",
    "Terraza": "Terraza",
    "Piscina": "Piscina",
    "Garaje": "Garaje",
}

_COLUMN_ORDER = [
    ("Compra",   "linear", "Sales (Linear)"),
    ("Compra",   "log",    "Sales (Log)"),
    ("Alquiler", "linear", "Rental (Linear)"),
    ("Alquiler", "log",    "Rental (Log)"),
]


def _extract_coef_rows(result, variables: list) -> dict:
    """Return {var: {coef, se, pval}} for every variable in ``variables``
    that appears in the fitted model's parameter index."""
    rows = {}
    for v in variables:
        if v in result.params.index:
            rows[v] = {
                "coef": float(result.params[v]),
                "se":   float(result.bse[v]),
                "pval": float(result.pvalues[v]),
            }
    return rows


def _stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def _csv_safe(d: dict) -> dict:
    """Strip nested objects (e.g. coef_rows) before serializing to CSV."""
    return {k: v for k, v in d.items() if k != "coef_rows"}


def _write_hedonic_combined_table(results: dict) -> None:
    """Single 4-column hedonic table across (Compra, Alquiler) x (linear, log).
    """
    cells = {}
    for lab, spec, _ in _COLUMN_ORDER:
        if lab in results and spec in results[lab]:
            cells[(lab, spec)] = results[lab][spec]
    if not cells:
        print("[hedonic] No results available to build combined table.")
        return

    headers = ["Variable"] + [name for (_, _, name) in _COLUMN_ORDER]
    md = []
    md.append("| " + " | ".join(headers) + " |")
    md.append("| " + " | ".join(["---"] * len(headers)) + " |")

    # Coefficient block: two physical rows per variable (coef, then (se)).
    for var in KEY_COEF_VARS:
        display = _VAR_DISPLAY[var]
        is_treat = (var == cfg.TREATMENT)
        name_cell = f"**{display}**" if is_treat else display

        coef_cells, se_cells = [], []
        for lab, spec, _ in _COLUMN_ORDER:
            row = cells.get((lab, spec), {}).get("coef_rows", {}).get(var)
            if row is None:
                coef_cells.append("—")
                se_cells.append("")
            else:
                coef_cells.append(f"{row['coef']:.4f}{_stars(row['pval'])}")
                se_cells.append(f"({row['se']:.4f})")
        md.append("| " + " | ".join([name_cell] + coef_cells) + " |")
        md.append("| " + " | ".join([""] + se_cells) + " |")

    # Footer separator
    md.append("| " + " | ".join(["---"] * len(headers)) + " |")

    def _footer(label, key, fmt):
        vals = []
        for lab, spec, _ in _COLUMN_ORDER:
            c = cells.get((lab, spec))
            vals.append(fmt(c[key]) if c is not None else "—")
        md.append("| " + " | ".join([label] + vals) + " |")

    _footer("Adjusted R²", "r2_adj", lambda x: f"{x:.4f}")
    _footer("AIC",              "aic",    lambda x: f"{x:,.0f}")
    _footer("N",                "nobs",   lambda x: f"{x:,}")

    n_cols = len(_COLUMN_ORDER)
    md.append("| " + " | ".join(["Municipality FE"]  + ["Yes"] * n_cols) + " |")
    md.append("| " + " | ".join(["Property-type FE"] + ["Yes"] * n_cols) + " |")
    md.append("| " + " | ".join(["Cluster-SE level"] + [cfg.FE_VAR] * n_cols) + " |")

    md.append("")
    md.append("Notes: Standard errors clustered at the municipality "
              f"({cfg.FE_VAR}) level in parentheses. "
              "Significance: *** p<0.01, ** p<0.05, * p<0.10. "
              "Em-dash (—) marks variables not present in the given "
              "specification (Metros/Metros² in log spec; log(Metros) "
              "in linear spec).")

    out_md = cfg.MODEL_DIR / "hedonic_combined_table.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"  [table] {out_md}")

    long_rows = []
    for lab, spec, name in _COLUMN_ORDER:
        c = cells.get((lab, spec))
        if c is None:
            continue
        for var in KEY_COEF_VARS:
            row = c.get("coef_rows", {}).get(var)
            if row is None:
                continue
            long_rows.append({
                "subsample": lab,
                "spec": spec,
                "column": name,
                "variable": var,
                "coef": row["coef"],
                "se": row["se"],
                "pval": row["pval"],
                "stars": _stars(row["pval"]),
            })
    out_csv = cfg.MODEL_DIR / "hedonic_combined_coefs.csv"
    pd.DataFrame(long_rows).to_csv(out_csv, index=False)
    print(f"  [table] {out_csv}")
 
 
def _vif_table(
    df: pd.DataFrame,
    cols: list,
) -> pd.DataFrame:
    """Compute Variance Inflation Factors for the requested columns."""
    sub = df[cols].dropna()
    return pd.DataFrame({
        "Variable": cols,
        "VIF": [variance_inflation_factor(sub.values, i)
                for i in range(sub.shape[1])],
    })

 
def _plot_residuals(result, label: str, spec: str) -> None:
    fitted = result.fittedvalues
    resid = result.resid
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Residual diagnostics - {label} ({spec})", fontsize=12)
 
    axes[0].scatter(fitted, resid, alpha=0.05, s=1, color="steelblue")
    axes[0].axhline(0, color="red", linewidth=1)
    axes[0].set_xlabel("Fitted values"); axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs fitted")
 
    axes[1].hist(resid, bins=80, color="steelblue", alpha=0.7, edgecolor="none")
    axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual distribution")
 
    stats.probplot(resid, dist="norm", plot=axes[2])
    axes[2].set_title("Q-Q normal")
 
    fig.tight_layout()
    out = cfg.FIG_DIR / f"hedonic_residuals_{label.lower()}_{spec}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {out}")
 
 
def _plot_coef_comparison(results: dict) -> None:
    """Compare the agency coefficient across the four (label, spec) cells."""
    cells = [(lab, spec) for lab in ("Compra", "Alquiler")
                          for spec in ("linear", "log")]
    coefs = [results[lab][spec]["coef"]  for lab, spec in cells]
    ci_lo = [results[lab][spec]["ci_lo"] for lab, spec in cells]
    ci_hi = [results[lab][spec]["ci_hi"] for lab, spec in cells]
    err_lo = [c - lo for c, lo in zip(coefs, ci_lo)]
    err_hi = [hi - c for c, hi in zip(coefs, ci_hi)]
    labels = [f"{lab}\n({spec})" for lab, spec in cells]
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Agency coefficient - comparison across specifications")
    for ax, idx, title in zip(
        axes, [[0, 2], [1, 3]],
        ["Compra (sale)", "Alquiler (rental)"],
    ):
        sub_lab = [labels[i] for i in idx]
        sub_co = [coefs[i]  for i in idx]
        sub_lo = [err_lo[i] for i in idx]
        sub_hi = [err_hi[i] for i in idx]
        colors = ["steelblue" if c > 0 else "tomato" for c in sub_co]
        ax.bar(sub_lab, sub_co, color=colors, width=0.5, alpha=0.85)
        ax.errorbar(sub_lab, sub_co, yerr=[sub_lo, sub_hi],
                    fmt="none", color="black", capsize=5)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(title); ax.set_ylabel("Estimated coefficient")
        for i, c in enumerate(sub_co):
            ax.text(i, c, f"{c:.4f}", ha="center", va="bottom", fontsize=9)
 
    fig.tight_layout()
    out = cfg.FIG_DIR / "hedonic_coef_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {out}")

 
def fit_one(
    df: pd.DataFrame,
    label: str,
    *,
    base_numeric: list = None,
    dummies: list = None,
    categorical_vars: list = None,
    treatment: str = cfg.TREATMENT,
    fe_var: str = cfg.FE_VAR,
) -> dict:
    """Estimate linear and log-linear hedonic regressions on one subsample.
    """
    base_numeric = base_numeric or [c for c in cfg.NUMERIC_FEATURES if c != "Metros"]
    dummies = dummies or cfg.DUMMY_FEATURES
    categorical_vars = categorical_vars or cfg.CATEGORICAL_FEATURES

    size_cols = ["Metros", "Metros2", "log_Metros"]
    needed = (base_numeric + size_cols + dummies + categorical_vars
              + [treatment, fe_var, cfg.TIME_FE_VAR, cfg.TARGET_LIN])
    sub = df[[c for c in needed if c in df.columns]].dropna().copy()
    sub = ft.trim_outliers(sub, cols=("Precio", "Metros"))
    sub = ft.filter_thin_municipalities(sub)
    sub[cfg.TARGET_LOG] = np.log(sub[cfg.TARGET_LIN])

    print(f"\n{'-'*60}\n  HEDONIC OLS - {label}\n{'-'*60}")
    print(f"  Observations (trimmed, thin municipalities removed): {len(sub):,}")
    if categorical_vars:
        for c in categorical_vars:
            if c in sub.columns:
                print(f"  Levels in C({c}): {sub[c].nunique()}")

    # VIF on the numeric predictor block.
    vif_cols = ["log_Metros"] + base_numeric + dummies + [treatment]
    vif_cols = [c for c in vif_cols if c in sub.columns]
    vif = _vif_table(sub, vif_cols)
    print("\n  Variance Inflation Factors (log-spec numerics):")
    print(vif.to_string(index=False))
    vif.to_csv(cfg.MODEL_DIR / f"hedonic_vif_{label.lower()}.csv", index=False)

    time_fe = cfg.TIME_FE_VAR if cfg.TIME_FE_VAR in sub.columns else None
    out = {}
    for spec, target in [("linear", cfg.TARGET_LIN), ("log", cfg.TARGET_LOG)]:
        formula = _build_formula(
            target,
            spec=spec,
            base_numeric=base_numeric, dummies=dummies,
            treatment=treatment, fe_var=fe_var,
            categorical_vars=categorical_vars,
            time_fe_var=time_fe,
        )
        print(f"\n  Estimating {spec} model (cluster-SE at {fe_var})...")
        # Standard errors clustered at the municipality level.
        result = smf.ols(formula, data=sub).fit(
            cov_type = "cluster",
            cov_kwds = {"groups": sub[fe_var]},
        )
        out[spec] = _extract_treatment_results(result, treatment)
        out[spec]["coef_rows"] = _extract_coef_rows(result, KEY_COEF_VARS)
        _plot_residuals(result, label, spec)
 
    # Comparative summary block
    lin, log = out["linear"], out["log"]
    pct = (np.exp(log["coef"]) - 1) * 100
    print(f"\n  {'':22s} {'Linear':>14s} {'Log-linear':>14s}")
    print(f"  {'-'*52}")
    print(f"  {'Agency coef':22s} {lin['coef']:>14.4f} {log['coef']:>14.4f}")
    print(f"  {'Std error':22s} {lin['se']:>14.4f} {log['se']:>14.4f}")
    print(f"  {'p-value':22s} {lin['pval']:>14.4f} {log['pval']:>14.4f}")
    print(f"  {'95% CI low':22s} {lin['ci_lo']:>14.4f} {log['ci_lo']:>14.4f}")
    print(f"  {'95% CI high':22s} {lin['ci_hi']:>14.4f} {log['ci_hi']:>14.4f}")
    print(f"  {'Adj. R^2':22s} {lin['r2_adj']:>14.4f} {log['r2_adj']:>14.4f}")
    print(f"  {'AIC':22s} {lin['aic']:>14.0f} {log['aic']:>14.0f}")
    print(f"  {'N':22s} {lin['nobs']:>14,d} {log['nobs']:>14,d}")
    print(f"\n  Log-linear interpretation: exp(beta) - 1 = {pct:+.2f}%")
    print(f"  -> Conditional on observables, an agency listing is")
    print(f"     associated with a price {abs(pct):.2f}% "
          f"{'higher' if pct > 0 else 'lower'} than a private listing.")
 
    rows = [
        {"Submuestra": label, "Especificacion": "linear", **_csv_safe(lin)},
        {"Submuestra": label, "Especificacion": "log",    **_csv_safe(log)},
    ]
    pd.DataFrame(rows).to_csv(
        cfg.MODEL_DIR / f"hedonic_results_{label.lower()}.csv", index=False
    )
    return out
 
 
def run(df: pd.DataFrame) -> dict:
    """Estimate hedonic OLS on Compra and Alquiler subsamples."""
    results = {}
    for label in ("Compra", "Alquiler"):
        df_op = df[df["Operacion"] == label].copy()
        if df_op.empty:
            print(f"[hedonic] No rows for operation = {label}; skipping.")
            continue
        results[label] = fit_one(df_op, label)
 
    if len(results) == 2:
        _plot_coef_comparison(results)
        _write_hedonic_combined_table(results)

    # Combined CSV for the appendix
    combined = []
    for label, res in results.items():
        for spec, vals in res.items():
            combined.append({
                "Submuestra": label,
                "Especificacion": spec,
                **_csv_safe(vals),
            })
    pd.DataFrame(combined).to_csv(
        cfg.MODEL_DIR / "hedonic_results_all.csv", index=False
    )
    return results