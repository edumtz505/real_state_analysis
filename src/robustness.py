import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import config as cfg
import features as ft

def _prep(df: pd.DataFrame, lower=0.01, upper=0.99) -> pd.DataFrame:
    """Sample preparation parallel to hedonic.fit_one but with configurable
    outlier trim. ``lower=None`` skips trimming entirely."""
    needed = (cfg.NUMERIC_FEATURES + cfg.ENGINEERED_NUMERIC + cfg.DUMMY_FEATURES
              + cfg.CATEGORICAL_FEATURES
              + [cfg.TREATMENT, cfg.FE_VAR, "NCA",
                 cfg.TIME_FE_VAR, cfg.TARGET_LIN])
    sub = df[[c for c in needed if c in df.columns]].dropna().copy()
    if lower is not None:
        sub = ft.trim_outliers(sub, cols=("Precio", "Metros"),
                                lower=lower, upper=upper)
    sub = ft.filter_thin_municipalities(sub)
    sub[cfg.TARGET_LOG] = np.log(sub[cfg.TARGET_LIN])
    return sub

def _fit_one(formula: str, data: pd.DataFrame,
             cov_type: str, cluster_col: str | None):
    if cov_type == "cluster":
        return smf.ols(formula, data=data).fit(
            cov_type="cluster",
            cov_kwds={"groups": data[cluster_col]},
        )
    return smf.ols(formula, data=data).fit(cov_type=cov_type)

def _extract_treatment(result, treatment: str = cfg.TREATMENT) -> dict:
    ci = result.conf_int().loc[treatment]
    coef = float(result.params[treatment])
    se   = float(result.bse[treatment])
    return {
        "agency_log": coef,
        "agency_pct": float((np.exp(coef) - 1) * 100),
        "se": se,
        "tstat": float(result.tvalues[treatment]),
        "pval": float(result.pvalues[treatment]),
        "ci_lo_pct": float((np.exp(ci[0]) - 1) * 100),
        "ci_hi_pct": float((np.exp(ci[1]) - 1) * 100),
        "r2_adj": float(result.rsquared_adj),
        "nobs": int(result.nobs),
    }

def _plot_forest(df: pd.DataFrame, label: str) -> None:
    """Forest plot of the agency premium across specifications."""
    if df.empty or "agency_pct" not in df.columns:
        return
    df = df.dropna(subset=["agency_pct"]).reset_index(drop=True)
    n = len(df)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * n + 1.5)))
    y = np.arange(n)[::-1]   # spec 1 at the top of the figure
    err_lo = (df["agency_pct"] - df["ci_lo_pct"]).clip(lower=0)
    err_hi = (df["ci_hi_pct"] - df["agency_pct"]).clip(lower=0)
    ax.errorbar(
        df["agency_pct"], y, xerr=[err_lo, err_hi],
        fmt="o", color="steelblue", ecolor="steelblue",
        capsize=4, markersize=7, linewidth=1.2,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["spec"].tolist(), fontsize=9)
    ax.set_xlabel("Agency premium (%) - log spec, exp(β)-1")
    ax.set_title(f"Agency coefficient - sensitivity across specifications ({label})")
    fig.tight_layout()
    out = cfg.FIG_DIR / f"robustness_forest_{label.lower()}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {out}")

def run_robustness(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Run the OLS specifications."""
    print(f"\n{'='*60}\n  ROBUSTNESS - agency coefficient sensitivity ({label})\n{'='*60}")

    # Three sample preparations
    sub_default = _prep(df, lower=0.01, upper=0.99)
    sub_loose = _prep(df, lower=0.005, upper=0.995)
    sub_notrim = _prep(df, lower=None, upper=None)
    print(f"  N (1/99 trim): {len(sub_default):,}")
    print(f"  N (0.5/99.5 trim):{len(sub_loose):,}")
    print(f"  N (no trim): {len(sub_notrim):,}")

    full_rhs = ("log_Metros + Habitaciones + Aseos"
                " + Terraza + Piscina + Garaje + Inmobiliaria"
                f" + C(Caracteristicas) + C({cfg.TIME_FE_VAR})")
    notype_rhs = ("log_Metros + Habitaciones + Aseos"
                  " + Terraza + Piscina + Garaje + Inmobiliaria"
                  f" + C({cfg.TIME_FE_VAR})")

    specs = [
        # (name, data, formula, cov_type, cluster_col)
        ("(1) Baseline: NMUN FE, cluster-SE, 1/99 trim",
         sub_default,
         f"log_Precio ~ {full_rhs} + C({cfg.FE_VAR})",
         "cluster", cfg.FE_VAR),

        ("(2) HC3 SE (vs cluster)",
         sub_default,
         f"log_Precio ~ {full_rhs} + C({cfg.FE_VAR})",
         "HC3", None),

        ("(3) Drop C(Caracteristicas) - no property-type FE",
         sub_default,
         f"log_Precio ~ {notype_rhs} + C({cfg.FE_VAR})",
         "cluster", cfg.FE_VAR),

        ("(4) Looser outlier trim (0.5/99.5)",
         sub_loose,
         f"log_Precio ~ {full_rhs} + C({cfg.FE_VAR})",
         "cluster", cfg.FE_VAR),

        ("(5) No outlier trim",
         sub_notrim,
         f"log_Precio ~ {full_rhs} + C({cfg.FE_VAR})",
         "cluster", cfg.FE_VAR),

        ("(6) NCA FE (coarser geography)",
         sub_default,
         f"log_Precio ~ {full_rhs} + C(NCA)",
         "cluster", "NCA"),
    ]

    rows = []
    for spec_name, data, formula, cov_type, cluster_col in specs:
        print(f"\n  >> {spec_name}  (N={len(data):,})")
        try:
            result = _fit_one(formula, data, cov_type, cluster_col)
            rec = _extract_treatment(result)
            rec["spec"] = spec_name
            rec["cov_type"] = cov_type
            rec["cluster_col"] = cluster_col if cluster_col else "-"
            rows.append(rec)
            print(f"    coef = {rec['agency_log']:+.4f}  "
                  f"({rec['agency_pct']:+.2f}%)  "
                  f"CI [{rec['ci_lo_pct']:+.2f}%, {rec['ci_hi_pct']:+.2f}%]  "
                  f"R2_adj={rec['r2_adj']:.4f}  N={rec['nobs']:,}")
        except Exception as e:
            print(f"    [error] {e}")
            rows.append({"spec": spec_name, "error": str(e)})

    out = pd.DataFrame(rows)
    # Move the spec column first for readability
    cols = (["spec"] + [c for c in out.columns if c != "spec"])
    out = out[cols]
    out_path = cfg.MODEL_DIR / f"robustness_specs_{label.lower()}.csv"
    out.to_csv(out_path, index=False)
    print(f"\n  [csv] {out_path}")

    _plot_forest(out, label)

    # Short interpretive summary
    valid = out.dropna(subset=["agency_pct"])
    if len(valid):
        rng_lo = valid["agency_pct"].min()
        rng_hi = valid["agency_pct"].max()
        print(f"\n  Agency premium range across {len(valid)} specs: "
              f"[{rng_lo:+.2f}%, {rng_hi:+.2f}%]")
        all_neg = (valid["ci_hi_pct"] < 0).all()
        all_pos = (valid["ci_lo_pct"] > 0).all()
        all_signif_same_sign = all_neg or all_pos
        any_cross_zero = ((valid["ci_lo_pct"] < 0) & (valid["ci_hi_pct"] > 0)).any()
        if all_signif_same_sign:
            sign = "negative" if all_neg else "positive"
            print(f"  All specifications: 95% CI strictly {sign} - "
                  "robustness confirmed.")
        elif any_cross_zero:
            n_cross = int(((valid["ci_lo_pct"] < 0)
                           & (valid["ci_hi_pct"] > 0)).sum())
            print(f"  {n_cross}/{len(valid)} specs have 95% CI crossing zero.")
    return out

def run(df: pd.DataFrame) -> dict:
    """Run robustness on Compra and Alquiler."""
    out = {}
    for label in ("Compra", "Alquiler"):
        df_op = df[df["Operacion"] == label].copy()
        if df_op.empty:
            print(f"[robustness] No rows for {label}; skipping.")
            continue
        out[label] = run_robustness(df_op, label)
        run_missingness_sensitivity(df_op, label)
    return out


_IMPUTE_COLS = ["log_Metros", "Habitaciones", "Aseos",
                "Terraza", "Piscina", "Garaje"]


def _prep_with_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Sample preparation parallel to hedonic.fit_one but with configurable
    outlier trim, and without dropping rows with missing numeric
    features. ``lower=None`` skips trimming entirely."""
    needed = (cfg.NUMERIC_FEATURES + cfg.ENGINEERED_NUMERIC + cfg.DUMMY_FEATURES
              + cfg.CATEGORICAL_FEATURES
              + [cfg.TREATMENT, cfg.FE_VAR, "NCA",
                 cfg.TIME_FE_VAR, cfg.TARGET_LIN])
    sub = df[[c for c in needed if c in df.columns]].copy()
    essential = [c for c in (cfg.TARGET_LIN, cfg.TREATMENT, cfg.FE_VAR,
                             cfg.TIME_FE_VAR, "Caracteristicas")
                 if c in sub.columns]
    sub = sub.dropna(subset=essential)
    sub = ft.trim_outliers(sub, cols=("Precio",))
    sub = ft.filter_thin_municipalities(sub)
    sub[cfg.TARGET_LOG] = np.log(sub[cfg.TARGET_LIN])
    return sub


def _fit_clustered(formula: str, data: pd.DataFrame):
    return smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data[cfg.FE_VAR]},
    )


def _fit_listwise(sub: pd.DataFrame, formula: str):
    """(A) Drop rows with any missing imputation-target column."""
    impute_cols = [c for c in _IMPUTE_COLS if c in sub.columns]
    data = sub.dropna(subset=impute_cols).copy()
    return _fit_clustered(formula, data), len(data)


def _fit_median_indicator(sub: pd.DataFrame, formula: str):
    """(B) Median imputation + missing-indicator dummies appended."""
    data = sub.copy()
    ind_terms = []
    for col in _IMPUTE_COLS:
        if col not in data.columns:
            continue
        was_miss = data[col].isna()
        data[col] = data[col].fillna(data[col].median())
        ind = f"{col}_was_missing"
        data[ind] = was_miss.astype(int)
        if was_miss.any():
            ind_terms.append(ind)
    full_formula = (formula + (" + " + " + ".join(ind_terms))) if ind_terms else formula
    return _fit_clustered(full_formula, data), len(data)


def _fit_multiple_imputation(sub: pd.DataFrame, formula: str,
                              m: int = 5, seed: int = cfg.SEED):
    """(C) M draws of IterativeImputer(sample_posterior=True); refit the
    OLS on each completed dataset."""
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    impute_cols = [c for c in _IMPUTE_COLS if c in sub.columns]
    coefs, var_within = [], []
    for i in range(m):
        imp = IterativeImputer(
            sample_posterior=True, random_state=seed + i, max_iter=10,
        )
        data = sub.copy()
        data[impute_cols] = imp.fit_transform(data[impute_cols])
        res = _fit_clustered(formula, data)
        coefs.append(float(res.params[cfg.TREATMENT]))
        var_within.append(float(res.bse[cfg.TREATMENT]) ** 2)
    coefs = np.array(coefs)
    var_within = np.array(var_within)

    # Rubin's rules
    beta_bar = float(coefs.mean())
    U = float(var_within.mean())
    B = float(coefs.var(ddof=1)) if m > 1 else 0.0
    T = U + (1.0 + 1.0 / m) * B
    se = float(np.sqrt(T))

    if B > 0:
        r = (1.0 + 1.0 / m) * B / U
        dof = float((m - 1) * (1.0 + 1.0 / r) ** 2)
    else:
        dof = float("inf")
    return beta_bar, se, dof, len(sub), m


def run_missingness_sensitivity(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Compare the agency coefficient under three missing-data strategies:
    (A) listwise deletion, (B) median + missing-indicator, (C) multiple
    imputation (IterativeImputer, sample_posterior=True). 
    Write a CSV side-by-side with the spec robustness table."""
    print(f"\n{'='*60}\n  MISSINGNESS SENSITIVITY ({label})\n{'='*60}")
    sub = _prep_with_missing(df)
    impute_cols = [c for c in _IMPUTE_COLS if c in sub.columns]
    n_any_miss = int(sub[impute_cols].isna().any(axis=1).sum())
    print(f"  N (post-trim, pre-imputation): {len(sub):,}")
    print(f"  Rows with >=1 missing numeric feature: {n_any_miss:,} "
          f"({100 * n_any_miss / max(len(sub), 1):.2f}%)")

    rhs = ("log_Metros + Habitaciones + Aseos"
           " + Terraza + Piscina + Garaje + Inmobiliaria"
           f" + C(Caracteristicas) + C({cfg.TIME_FE_VAR})"
           f" + C({cfg.FE_VAR})")
    formula = f"log_Precio ~ {rhs}"

    rows = []

    print("\n  >> (A) Listwise deletion")
    res_a, n_a = _fit_listwise(sub, formula)
    rows.append({
        "strategy": "(A) Listwise deletion",
        "agency_log": float(res_a.params[cfg.TREATMENT]),
        "se": float(res_a.bse[cfg.TREATMENT]),
        "nobs": n_a,
        "M": None,
        "dof": float(res_a.df_resid),
    })

    print("  >> (B) Median + missing-indicator")
    res_b, n_b = _fit_median_indicator(sub, formula)
    rows.append({
        "strategy": "(B) Median + missing-indicator",
        "agency_log": float(res_b.params[cfg.TREATMENT]),
        "se": float(res_b.bse[cfg.TREATMENT]),
        "nobs": n_b,
        "M": None,
        "dof": float(res_b.df_resid),
    })

    print("  >> (C) Multiple imputation (Rubin's rules)")
    beta, se, dof, n_c, m_c = _fit_multiple_imputation(sub, formula, m=5)
    rows.append({
        "strategy": "(C) Multiple imputation",
        "agency_log": beta,
        "se": se,
        "nobs": n_c,
        "M": m_c,
        "dof": dof,
    })

    for r in rows:
        lo = r["agency_log"] - 1.96 * r["se"]
        hi = r["agency_log"] + 1.96 * r["se"]
        r["agency_pct"] = float((np.exp(r["agency_log"]) - 1) * 100)
        r["ci_lo_pct"] = float((np.exp(lo) - 1) * 100)
        r["ci_hi_pct"] = float((np.exp(hi) - 1) * 100)

    out = pd.DataFrame(rows)[
        ["strategy", "agency_log", "se", "agency_pct",
         "ci_lo_pct", "ci_hi_pct", "nobs", "M", "dof"]
    ]
    print()
    print(out.to_string(index=False))

    out_path = cfg.MODEL_DIR / f"robustness_missing_{label.lower()}.csv"
    out.to_csv(out_path, index=False)
    print(f"\n  [csv] {out_path}")

    spread = out["agency_pct"].max() - out["agency_pct"].min()
    print(f"  Spread across (A)/(B)/(C): {spread:+.2f} pp on the % scale.")
    return out
