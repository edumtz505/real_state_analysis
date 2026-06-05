import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import config as cfg
import features as ft

warnings.filterwarnings("ignore")


def _prep_sub(df_op: pd.DataFrame) -> pd.DataFrame:
    """Trim + thin-municipality filter so coefficients
    are comparable across modules."""
    needed = (cfg.NUMERIC_FEATURES + cfg.ENGINEERED_NUMERIC + cfg.DUMMY_FEATURES
              + cfg.CATEGORICAL_FEATURES + [cfg.TREATMENT, cfg.FE_VAR, "NCA",
              cfg.TIME_FE_VAR, cfg.TARGET_LIN])
    sub = df_op[[c for c in needed if c in df_op.columns]].dropna().copy()
    sub = ft.trim_outliers(sub, cols=("Precio", "Metros"))
    sub = ft.filter_thin_municipalities(sub)
    sub[cfg.TARGET_LOG] = np.log(sub[cfg.TARGET_LIN])
    return sub


def _fit_clustered(formula: str, data: pd.DataFrame, cluster_col: str):
    """OLS with municipality-clustered SE."""
    return smf.ols(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data[cluster_col]},
    )


def _fit_hc3(formula: str, data: pd.DataFrame):
    """Same OLS, but with HC3 SE. Used as a fallback when the cluster
    covariance for an interaction term is singular (too few clusters within a
    subgroup for the within-cluster correlation structure to be identified)."""
    return smf.ols(formula, data=data).fit(cov_type="HC3")


def _marginal_se(result_main, result_fallback, main: str, ix_term: str,
                 sing_threshold: float = 1.0) -> tuple[float, str]:
    """Compute the SE of (beta_main + beta_ix) under the primary cov estimator.
    """
    def _se_from(res) -> float:
        cov = res.cov_params()
        if main not in cov.index or ix_term not in cov.index:
            return np.nan
        var = (cov.loc[main, main]
               + cov.loc[ix_term, ix_term]
               + 2 * cov.loc[main, ix_term])
        if not np.isfinite(var) or var < 0:
            return np.nan
        return float(np.sqrt(var))

    se_main = _se_from(result_main)
    if np.isfinite(se_main) and 0 < se_main <= sing_threshold:
        return se_main, "cluster"
    se_fb = _se_from(result_fallback)
    if np.isfinite(se_fb):
        return se_fb, "hc3"
    return np.nan, "failed"


def _filter_rare_buckets(sub: pd.DataFrame, col: str,
                         min_n: int = cfg.MIN_N_PER_HET_BUCKET) -> pd.DataFrame:
    """Drop rows in subgroups smaller than ``min_n``."""
    counts = sub[col].value_counts()
    keepers = counts[counts >= min_n].index
    dropped = counts[counts < min_n]
    if len(dropped):
        print(f"  Dropping {len(dropped)} rare {col} levels "
              f"(< {min_n} listings each, N_total={int(dropped.sum()):,}):")
        for k, v in dropped.items():
            print(f"    - {k}: {v}")
    return sub[sub[col].isin(keepers)].copy()


def _is_degenerate(coef: float, se: float,
                   coef_max: float = 2.0, se_max: float = 1.0) -> bool:
    # Some subgroup interactions came back with absurd numbers (mostly inf)
    # when the bucket had too few clusters and the cluster-SE matrix was nearly
    # singular. The caps below catch those rows so they don't end up in the
    # output table.
    return (
        not np.isfinite(coef) or not np.isfinite(se)
        or abs(coef) > coef_max or se > se_max or se <= 0
    )


def _coef_table(result, term_filter: str = None) -> pd.DataFrame:
    """Pull a tidy frame of (term, coef, se, t, p, ci_lo, ci_hi) from a fit."""
    summary = pd.DataFrame({
        "coef": result.params,
        "se": result.bse,
        "tstat": result.tvalues,
        "pval": result.pvalues,
    })
    ci = result.conf_int()
    summary["ci_lo"] = ci[0]
    summary["ci_hi"] = ci[1]
    summary = summary.reset_index().rename(columns={"index": "term"})
    if term_filter:
        summary = summary[summary["term"].str.contains(term_filter, regex=True)]
    return summary


def run_caracteristicas_interaction(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Estimate log(Precio) ~ Inm * C(Caracteristicas) + controls + FE.
    """
    print(f"\n{'-'*60}\n  HETEROGENEITY 1 - agency × property type ({label})\n{'-'*60}")
    sub = _prep_sub(df)
    sub = _filter_rare_buckets(sub, "Caracteristicas")

    formula = (
        "log_Precio ~ Inmobiliaria * C(Caracteristicas)"
        " + log_Metros + Habitaciones + Aseos"
        " + Terraza + Piscina + Garaje"
        f" + C({cfg.FE_VAR}) + C({cfg.TIME_FE_VAR})"
    )
    res_cluster = _fit_clustered(formula, sub, cluster_col=cfg.FE_VAR)
    res_hc3 = _fit_hc3(formula, sub)

    # Compute marginal agency premium per property type
    params = res_cluster.params
    main = "Inmobiliaria"
    types = sorted(sub["Caracteristicas"].unique())

    rows = []
    dropped_degen = []
    for t in types:
        ix_term = f"Inmobiliaria:C(Caracteristicas)[T.{t}]"
        if t == types[0]:
            # Baseline level: marginal effect is the main coefficient itself.
            coef = float(params[main])
            se_main = float(res_cluster.bse[main])
            se_fb   = float(res_hc3.bse[main])
            if np.isfinite(se_main) and 0 < se_main <= 1.0:
                se, method = se_main, "cluster"
            else:
                se, method = se_fb, "hc3"
        else:
            if ix_term not in params.index:
                continue
            coef = float(params[main] + params[ix_term])
            se, method = _marginal_se(res_cluster, res_hc3, main, ix_term)

        # Drop numerically degenerate rows
        if _is_degenerate(coef, se):
            dropped_degen.append((t, coef, se))
            continue

        bucket = sub[sub["Caracteristicas"] == t]
        n_t = int(len(bucket))
        n_clusts = int(bucket[cfg.FE_VAR].nunique())
        flag = "low_clusters" if n_clusts < 5 else ""
        rows.append({
            "Caracteristicas": t,
            "N": n_t,
            "n_clusters": n_clusts,
            "agency_coef": coef,
            "agency_pct": float((np.exp(coef) - 1) * 100),
            "se": se,
            "ci_lo_pct": float((np.exp(coef - 1.96 * se) - 1) * 100),
            "ci_hi_pct": float((np.exp(coef + 1.96 * se) - 1) * 100),
            "se_method": method,
            "flag": flag,
        })

    if dropped_degen:
        print(f"  Dropped {len(dropped_degen)} degenerate row(s) (interaction"
              f" collinear with FE - coef or SE outside plausible range):")
        for t, coef, se in dropped_degen:
            print(f"    - {t}: coef={coef:.2g}, se={se:.2g}")
    out = (pd.DataFrame(rows)
             .sort_values("agency_pct", ascending=False)
             .reset_index(drop=True))
    out_path = cfg.MODEL_DIR / f"heterogeneity_caracteristicas_{label.lower()}.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"  [csv] {out_path}")
    _plot_heterogeneity(out, x="Caracteristicas", label=label,
                        title=f"Agency premium by property type ({label})",
                        fname=f"hetero_caracteristicas_{label.lower()}")
    return out


def run_size_tertile_split(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Estimate the agency coefficient separately on small/medium/large
    listings.
    """
    print(f"\n{'-'*60}\n  HETEROGENEITY 2 - agency by size tertile ({label})\n{'-'*60}")
    sub = _prep_sub(df)

    # Global Metros tertile boundaries
    q33, q67 = sub["Metros"].quantile([1 / 3, 2 / 3]).values
    print(f"  Metros tertile breakpoints: q33={q33:.0f} m^2, q67={q67:.0f} m^2")

    def _bucket(m: float) -> str:
        if pd.isna(m):
            return None
        if m <= q33:
            return "Small"
        if m <= q67:
            return "Medium"
        return "Large"

    sub["Metros_tert"] = sub["Metros"].map(_bucket)
    sub = sub.dropna(subset=["Metros_tert"])

    rows = []
    for tert in ["Small", "Medium", "Large"]:
        s = sub[sub["Metros_tert"] == tert].copy()
        if s.empty:
            continue
        formula = (
            "log_Precio ~ Inmobiliaria + log_Metros + Habitaciones + Aseos"
            " + Terraza + Piscina + Garaje + C(Caracteristicas)"
            f" + C({cfg.FE_VAR}) + C({cfg.TIME_FE_VAR})"
        )
        res = _fit_clustered(formula, s, cluster_col=cfg.FE_VAR)
        coef = float(res.params["Inmobiliaria"])
        se = float(res.bse["Inmobiliaria"])
        rows.append({
            "Metros_tert": tert,
            "N": int(len(s)),
            "agency_coef": coef,
            "agency_pct": float((np.exp(coef) - 1) * 100),
            "se": se,
            "pval": float(res.pvalues["Inmobiliaria"]),
            "ci_lo_pct": float((np.exp(coef - 1.96 * se) - 1) * 100),
            "ci_hi_pct": float((np.exp(coef + 1.96 * se) - 1) * 100),
        })

    out = pd.DataFrame(rows)
    out_path = cfg.MODEL_DIR / f"heterogeneity_size_{label.lower()}.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"  [csv] {out_path}")
    _plot_heterogeneity(out, x="Metros_tert", label=label,
                        title=f"Agency premium by size tertile ({label})",
                        fname=f"hetero_size_{label.lower()}")
    return out


def run_region_interaction(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Estimate log(Precio) ~ Inm * C(NCA) + controls + FE.
    """
    print(f"\n{'-'*60}\n  HETEROGENEITY 3 - agency × region (NCA) ({label})\n{'-'*60}")
    sub = _prep_sub(df)
    if "NCA" not in sub.columns:
        print("  NCA column not available; skipping.")
        return pd.DataFrame()
    sub = _filter_rare_buckets(sub, "NCA")

    formula = (
        "log_Precio ~ Inmobiliaria * C(NCA)"
        " + log_Metros + Habitaciones + Aseos"
        " + Terraza + Piscina + Garaje + C(Caracteristicas)"
        f" + C({cfg.FE_VAR}) + C({cfg.TIME_FE_VAR})"
    )
    res_cluster = _fit_clustered(formula, sub, cluster_col=cfg.FE_VAR)
    res_hc3 = _fit_hc3(formula, sub)

    params = res_cluster.params
    main = "Inmobiliaria"
    regs = sorted(sub["NCA"].unique())

    rows = []
    dropped_degen = []
    for r in regs:
        ix_term = f"Inmobiliaria:C(NCA)[T.{r}]"
        if r == regs[0]:
            coef = float(params[main])
            se_main = float(res_cluster.bse[main])
            se_fb = float(res_hc3.bse[main])
            if np.isfinite(se_main) and 0 < se_main <= 1.0:
                se, method = se_main, "cluster"
            else:
                se, method = se_fb, "hc3"
        else:
            if ix_term not in params.index:
                continue
            coef = float(params[main] + params[ix_term])
            se, method = _marginal_se(res_cluster, res_hc3, main, ix_term)

        if _is_degenerate(coef, se):
            dropped_degen.append((r, coef, se))
            continue

        bucket = sub[sub["NCA"] == r]
        n_r  = int(len(bucket))
        n_clusts = int(bucket[cfg.FE_VAR].nunique())
        flag = "low_clusters" if n_clusts < 5 else ""
        rows.append({
            "NCA": r,
            "N": n_r,
            "n_clusters": n_clusts,
            "agency_coef": coef,
            "agency_pct": float((np.exp(coef) - 1) * 100),
            "se": se,
            "ci_lo_pct": float((np.exp(coef - 1.96 * se) - 1) * 100),
            "ci_hi_pct": float((np.exp(coef + 1.96 * se) - 1) * 100),
            "se_method": method,
            "flag": flag,
        })

    if dropped_degen:
        print(f"  Dropped {len(dropped_degen)} degenerate row(s):")
        for r, coef, se in dropped_degen:
            print(f"    - {r}: coef={coef:.2g}, se={se:.2g}")
    out = (pd.DataFrame(rows)
             .sort_values("agency_pct", ascending=False)
             .reset_index(drop=True))
    out_path = cfg.MODEL_DIR / f"heterogeneity_region_{label.lower()}.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"  [csv] {out_path}")
    _plot_heterogeneity(out, x="NCA", label=label,
                        title=f"Agency premium by region (NCA, {label})",
                        fname=f"hetero_region_{label.lower()}")
    return out


def _plot_heterogeneity(df: pd.DataFrame, *, x: str, label: str,
                        title: str, fname: str) -> None:
    if df.empty:
        return
    n = len(df)
    fig_h = max(4, 0.35 * n + 2)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    df_sorted = df.sort_values("agency_pct")
    colors = ["tomato" if v < 0 else "steelblue" for v in df_sorted["agency_pct"]]
    ax.barh(df_sorted[x].astype(str), df_sorted["agency_pct"],
            color=colors, alpha=0.85)
    err_lo = df_sorted["agency_pct"] - df_sorted["ci_lo_pct"]
    err_hi = df_sorted["ci_hi_pct"] - df_sorted["agency_pct"]
    ax.errorbar(df_sorted["agency_pct"], df_sorted[x].astype(str),
                xerr=[err_lo, err_hi], fmt="none", color="black",
                capsize=3, linewidth=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Agency premium (%)")
    ax.set_title(title)
    fig.tight_layout()
    out = cfg.FIG_DIR / f"{fname}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {out}")


def run(df: pd.DataFrame) -> dict:
    """Run all three heterogeneity analyses on Compra and Alquiler."""
    out = {"caracteristicas": {}, "size": {}, "region": {}}
    for label in ("Compra", "Alquiler"):
        df_op = df[df["Operacion"] == label].copy()
        if df_op.empty:
            print(f"[heterogeneity] No rows for {label}; skipping.")
            continue
        out["caracteristicas"][label] = run_caracteristicas_interaction(df_op, label)
        out["size"][label] = run_size_tertile_split(df_op, label)
        out["region"][label] = run_region_interaction(df_op, label)
    return out