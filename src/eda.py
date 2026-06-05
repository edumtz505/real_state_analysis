import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import config as cfg

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
 
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120})
 
_PCT_FMT = mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
 
 
def _save(fig: plt.Figure, name: str) -> Path:
    out = cfg.FIG_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {out}")
    return out

 
def describe_operation(df_op: pd.DataFrame, label: str) -> dict:
    """Print and save descriptive statistics for one operation type.
    """
    inm = df_op[df_op["Inmobiliaria"] == 1]
    par = df_op[df_op["Inmobiliaria"] == 0]
 
    print(f"\n{'#'*60}")
    print(f"#  ANALYSIS - {label.upper():^54}#")
    print(f"{'#'*60}")
    print(f"  Agency listings:     {len(inm):>9,}")
    print(f"  Private listings:    {len(par):>9,}")
    if "ClientID" in df_op.columns:
        print(f"  Unique agencies:     {inm['ClientID'].nunique():>9,}")
        print(f"  Unique private IDs:  {par['ClientID'].nunique():>9,}")
 
    # Descriptive statistics by group
    num_cols = [c for c in ["Precio", "Metros", "Habitaciones", "Aseos"]
                if c in df_op.columns]
    desc_full = df_op[num_cols].describe(percentiles=[.05, .25, .5, .75, .95]).T
    desc_inm = inm[num_cols].describe(percentiles=[.05, .25, .5, .75, .95]).T
    desc_par = par[num_cols].describe(percentiles=[.05, .25, .5, .75, .95]).T
 
    desc_full.to_csv(cfg.EDA_DIR / f"desc_{label.lower()}_total.csv")
    desc_inm.to_csv (cfg.EDA_DIR / f"desc_{label.lower()}_agency.csv")
    desc_par.to_csv (cfg.EDA_DIR / f"desc_{label.lower()}_private.csv")
 
    # Mann-Whitney for the unconditional price difference
    p_inm = inm["Precio"].dropna()
    p_par = par["Precio"].dropna()
    if len(p_inm) and len(p_par):
        stat, pval = stats.mannwhitneyu(p_inm, p_par, alternative="two-sided")
        effect_r = 2 * stat / (len(p_inm) * len(p_par)) - 1
        print(f"  Mann-Whitney U:      {stat:,.0f}   p-value: {pval:.4f}")
        print(f"  Effect size (r):     {effect_r:+.4f}  (|r|<0.1 small, 0.1–0.3 medium, >0.3 large)")
        print(f"  Median agency:       {p_inm.median():,.0f} EUR")
        print(f"  Median private:      {p_par.median():,.0f} EUR")
        diff_pct = (p_inm.median() - p_par.median()) / p_par.median() * 100
        print(f"  Median premium:      {diff_pct:+.2f}%")
 
    # Boxplot agency vs private
    p95 = df_op["Precio"].quantile(0.95)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=df_op[df_op["Precio"] <= p95],
        x="Tipo_Anunciante", y="Precio", ax=ax,
        order=["Inmobiliaria", "Particular"],
        palette=cfg.COLORES_INM,
    )
    ax.set_title(f"Price - Agency vs Private  ({label}, up to p95)")
    ax.set_ylabel("EUR"); ax.set_xlabel("")
    ax.yaxis.set_major_formatter(_PCT_FMT)
    _save(fig, f"box_price_{label.lower()}")
 
    # Overlaid histograms
    fig, ax = plt.subplots(figsize=(11, 5))
    for tipo, sub, c in [
        ("Inmobiliaria", inm, cfg.COLORES_INM["Inmobiliaria"]),
        ("Particular",   par, cfg.COLORES_INM["Particular"]),
    ]:
        data = sub["Precio"].dropna()
        p99 = data.quantile(0.99)
        ax.hist(data[data <= p99], bins=60, alpha=0.5, label=tipo,
                color=c, edgecolor="white", linewidth=0.3)
    ax.set_title(f"Price distribution by channel - {label}")
    ax.set_xlabel("EUR"); ax.set_ylabel("Frequency")
    ax.xaxis.set_major_formatter(_PCT_FMT)
    ax.legend()
    _save(fig, f"hist_price_{label.lower()}")
 
    return {
        "n_agency": len(inm),
        "n_private": len(par),
        "median_agency": float(p_inm.median()) if len(p_inm) else np.nan,
        "median_private": float(p_par.median()) if len(p_par) else np.nan,
        "mw_pvalue": float(pval) if len(p_inm) and len(p_par) else np.nan,
        "effect_r": float(effect_r) if len(p_inm) and len(p_par) else np.nan,
    }

 
def describe_municipalities(
    df_op: pd.DataFrame,
    label: str,
    *,
    top_n: int = 25,
    min_listings: int = cfg.MIN_LISTINGS_MUN,
    min_private: int = cfg.MIN_PRIVATE_MUN,
) -> pd.DataFrame:
    """Aggregate by municipality and rank by price, volume and agency premium."""
    counts = df_op["NMUN"].value_counts()
    keep = counts[counts >= min_listings].index
    df_mun = df_op[df_op["NMUN"].isin(keep)].copy()
 
    print(f"\n{'-'*60}")
    print(f"  MUNICIPAL ANALYSIS - {label.upper()}")
    print(f"  Municipalities with >= {min_listings} listings: {len(keep):,}")
    print(f"  Listings retained:                 {len(df_mun):,}"
          f"   ({len(df_mun)/len(df_op)*100:.1f}% of total)")
 
    summary = (
        df_mun.groupby("NMUN")
        .agg(
            N = ("Precio", "count"),
            Precio_mean = ("Precio", "mean"),
            Precio_median = ("Precio", "median"),
            Precio_std = ("Precio", "std"),
            Metros_med = ("Metros", "median"),
        )
        .round(2)
        .sort_values("N", ascending=False)
    )
    summary.to_csv(cfg.EDA_DIR / f"municipal_{label.lower()}_summary.csv")
 
    # Agency vs private median price by municipality
    grp = df_mun.groupby(["NMUN", "Tipo_Anunciante"])["Precio"]
    pivot = grp.median().unstack("Tipo_Anunciante").dropna(how="all")
    n_counts = grp.count().unstack("Tipo_Anunciante").fillna(0)

    if {"Inmobiliaria", "Particular"}.issubset(pivot.columns):
        # Enforce private-seller floor so thin groups don't distort the ranking
        valid_mun = n_counts[n_counts.get("Particular", 0) >= min_private].index
        pivot = pivot[pivot.index.isin(valid_mun)]
        print(f"  Municipalities with >= {min_private} private listings: {len(pivot):,}")

        pivot["N_agency"] = n_counts.loc[pivot.index, "Inmobiliaria"].astype(int)
        pivot["N_private"] = n_counts.loc[pivot.index, "Particular"].astype(int)
        pivot["Premium_%"] = (
            (pivot["Inmobiliaria"] - pivot["Particular"])
            / pivot["Particular"] * 100
        ).round(1)
        pivot = pivot.sort_values("Premium_%", ascending=False)
        pivot.to_csv(cfg.EDA_DIR / f"municipal_{label.lower()}_premium.csv")

        n_pos = int((pivot["Premium_%"] > 0).sum())
        n_neg = int((pivot["Premium_%"] <= 0).sum())
        print(f"  Municipalities w/ agency premium > 0:  {n_pos:,}")
        print(f"  Municipalities w/ agency premium <= 0: {n_neg:,}")
        print(f"  Mean municipal premium:    {pivot['Premium_%'].mean():.1f}%")
        print(f"  Median municipal premium:  {pivot['Premium_%'].median():.1f}%")
 
    # Mann-Whitney by municipality (only where both groups have >=10 obs)
    rows = []
    for mun in keep:
        sub  = df_mun[df_mun["NMUN"] == mun]
        g_in = sub.loc[sub["Inmobiliaria"] == 1, "Precio"].dropna()
        g_pa = sub.loc[sub["Inmobiliaria"] == 0, "Precio"].dropna()
        if len(g_in) >= 10 and len(g_pa) >= 10:
            stat, pval = stats.mannwhitneyu(g_in, g_pa, alternative="two-sided")
            rows.append({
                "NMUN": mun,
                "N_agency": len(g_in),
                "N_private": len(g_pa),
                "Median_agency": round(g_in.median(), 0),
                "Median_private": round(g_pa.median(), 0),
                "p_value": round(pval, 4),
                "Effect_r": round(2 * stat / (len(g_in) * len(g_pa)) - 1, 4),
                "Significant": pval < 0.05,
                "Agency_higher": g_in.median() > g_pa.median(),
            })
    tests = pd.DataFrame(rows)
    if len(tests):
        tests = tests.sort_values("p_value")
    tests.to_csv(cfg.EDA_DIR / f"municipal_{label.lower()}_mw.csv", index=False)
 
    if len(tests):
        n_sig = int(tests["Significant"].sum())
        n_sig_higher = int((tests["Significant"] & tests["Agency_higher"]).sum())
        print(f"  Municipalities tested:                 {len(tests):,}")
        print(f"  Significant differences (p<0.05):      {n_sig:,} "
              f"({n_sig/len(tests)*100:.1f}%)")
        print(f"  ... of which agency more expensive:    {n_sig_higher:,}")
 
    return summary
 

_CORR_VARS_CANDIDATE = [
    "Precio", "log_Precio", "PrecioM2",
    "Metros", "log_Metros",
    "Habitaciones", "Aseos",
    "Terraza", "Piscina", "Garaje",
    "n_amenities", "bath_per_room",
    "Inmobiliaria",
]

_NORMALITY_VARS_CANDIDATE = [
    "Precio", "log_Precio",
    "Metros", "log_Metros",
    "Habitaciones", "Aseos",
    "PrecioM2",
]

_NORM_SAMPLE_N = 5000
_SCATTER_SAMPLE_N = 10000


def _available(df: pd.DataFrame, candidates: list) -> list:
    return [c for c in candidates if c in df.columns]


def _rng(seed: int = cfg.SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def missing_data_report(df: pd.DataFrame) -> pd.DataFrame:
    """Column-level missingness for the full panel plus an agency/private split."""
    print(f"\n{'-'*60}\n  MISSING DATA REPORT (full panel)\n{'-'*60}")

    n = len(df)
    miss_total = df.isna().sum()
    pct_total = (miss_total / n * 100).round(2)
    out = pd.DataFrame({"n_missing": miss_total, "pct_missing": pct_total})

    # Compare missingness across the two channels
    if "Inmobiliaria" in df.columns:
        inm = df[df["Inmobiliaria"] == 1]
        par = df[df["Inmobiliaria"] == 0]
        out["pct_agency"]  = (inm.isna().sum() / max(len(inm), 1) * 100).round(2)
        out["pct_private"] = (par.isna().sum() / max(len(par), 1) * 100).round(2)
        out["pct_diff"] = (out["pct_agency"] - out["pct_private"]).round(2)

    out = out.sort_values("pct_missing", ascending=False)
    out_path = cfg.EDA_DIR / "ext_missing_summary.csv"
    out.to_csv(out_path)
    print(f"  [table] {out_path}")
    print(out.head(15).to_string())

    # Bar plot of overall missingness.
    top = out.head(25)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(top))))
    sns.barplot(x=top["pct_missing"], y=top.index, ax=ax, color="#4c72b0")
    ax.set_xlabel("% missing"); ax.set_ylabel("")
    ax.set_title("Missing values by column (top 25)")
    _save(fig, "ext_missing_bar")

    # Side-by-side comparison agency vs private.
    if {"pct_agency", "pct_private"}.issubset(out.columns):
        focus = _available(df, _CORR_VARS_CANDIDATE + ["NMUN", "CodigoPostal",
                                                       "Caracteristicas", "Fecha"])
        comp = out.loc[[c for c in focus if c in out.index],
                       ["pct_agency", "pct_private"]]
        comp = comp.sort_values("pct_agency", ascending=False)

        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(comp))))
        comp.plot(kind="barh", ax=ax,
                  color=[cfg.COLORES_INM["Inmobiliaria"],
                         cfg.COLORES_INM["Particular"]])
        ax.invert_yaxis()
        ax.set_xlabel("% missing"); ax.set_ylabel("")
        ax.set_title("Missingness by channel - modelling columns")
        ax.legend(["Agency", "Private"], loc = "lower right")
        _save(fig, "ext_missing_by_channel")

    return out


def correlation_matrix(df_op: pd.DataFrame, label: str) -> None:
    """Pearson and Spearman correlation matrices for the modelling features."""
    cols = _available(df_op, _CORR_VARS_CANDIDATE)
    sub = df_op[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if sub.shape[1] < 2 or len(sub) < 50:
        print(f"  [corr] Not enough numeric data for {label}; skipping.")
        return

    print(f"\n{'-'*60}\n  CORRELATION MATRIX - {label}\n{'-'*60}")
    print(f"  Variables ({sub.shape[1]}): {', '.join(sub.columns)}")
    print(f"  Observations used (complete cases): {len(sub):,}")

    for method, fname in [("pearson", "ext_corr_pearson"),
                          ("spearman", "ext_corr_spearman")]:
        corr = sub.corr(method=method)
        corr.round(4).to_csv(cfg.EDA_DIR / f"{fname}_{label.lower()}.csv")

        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        fig, ax = plt.subplots(figsize=(1.0 * len(corr) + 1.5,
                                        0.9 * len(corr) + 1.0))
        sns.heatmap(
            corr,
            mask = mask,
            cmap = "RdBu_r", center=0, vmin=-1, vmax=1,
            annot = True, fmt=".2f", annot_kws={"size": 8},
            square = True, linewidths=0.4, cbar_kws={"shrink": 0.75},
            ax = ax,
        )
        ax.set_title(f"{method.capitalize()} correlation - {label}")
        _save(fig, f"{fname}_{label.lower()}")

    # |Pearson - Spearman| as a non-linearity flag
    p = sub.corr(method="pearson")
    s = sub.corr(method="spearman")
    gap = (s - p).abs().round(4)
    gap.to_csv(cfg.EDA_DIR / f"ext_corr_gap_{label.lower()}.csv")


def _safe_anderson(x: np.ndarray) -> tuple[float, float]:
    """Anderson-Darling statistic and the *approximate* p-value via the
    5%-critical-value scaling. Returns (stat, p_approx).
    """
    try:
        res = stats.anderson(x, dist="norm")
        # Map A^2 to an approximate p-value using the 15% / 10% / 5% / 2.5% / 1%
        sig = res.significance_level / 100.0 
        cv = res.critical_values
        if res.statistic < cv[0]:
            return float(res.statistic), float(sig[0])
        for i in range(len(cv) - 1):
            if cv[i] <= res.statistic < cv[i + 1]:
                return float(res.statistic), float(sig[i + 1])
        return float(res.statistic), float(sig[-1])
    except Exception:
        return float("nan"), float("nan")


def normality_analysis(df_op: pd.DataFrame, label: str) -> pd.DataFrame:
    """Run a battery of normality diagnostics on every numeric variable.
    """
    print(f"\n{'-'*60}\n  NORMALITY ANALYSIS - {label}\n{'-'*60}")
    rng = _rng()
    rows = []
    var_cols = _available(df_op, _NORMALITY_VARS_CANDIDATE)

    for var in var_cols:
        x = pd.to_numeric(df_op[var], errors="coerce").dropna().to_numpy()
        n = x.size
        if n < 30:
            print(f"  {var:<14s}  n<30, skipped")
            continue

        skew = float(stats.skew(x, bias=False))
        kurt = float(stats.kurtosis(x, bias=False))  # excess kurtosis

        # Shapiro
        if n > _NORM_SAMPLE_N:
            idx = rng.choice(n, size=_NORM_SAMPLE_N, replace=False)
            xs  = x[idx]
        else:
            xs = x
        try:
            w_stat, w_p = stats.shapiro(xs)
        except Exception:
            w_stat, w_p = (np.nan, np.nan)

        # D'Agostino K^2
        try:
            k2_stat, k2_p = stats.normaltest(x)
        except Exception:
            k2_stat, k2_p = (np.nan, np.nan)

        # Anderson-Darling
        ad_stat, ad_p = _safe_anderson(x)

        # Jarque-Bera
        try:
            jb_stat, jb_p = stats.jarque_bera(x)
        except Exception:
            jb_stat, jb_p = (np.nan, np.nan)

        rows.append({
            "variable": var,
            "n": n,
            "mean": float(np.mean(x)),
            "std": float(np.std(x, ddof=1)),
            "skew": skew,
            "kurt_excess": kurt,
            "shapiro_W": float(w_stat) if not np.isnan(w_stat) else np.nan,
            "shapiro_p": float(w_p)    if not np.isnan(w_p)    else np.nan,
            "shapiro_n_used": int(xs.size),
            "dagostino_K2": float(k2_stat),
            "dagostino_p": float(k2_p),
            "anderson_A2": ad_stat,
            "anderson_p_approx": ad_p,
            "jarque_bera": float(jb_stat),
            "jarque_bera_p": float(jb_p),
            "normal_at_5pct": bool(
                (not np.isnan(w_p) and w_p > 0.05)
                and (not np.isnan(k2_p) and k2_p > 0.05)
            ),
        })

        # Figure: hist + KDE / boxplot / QQ
        plot_x = x if n <= _NORM_SAMPLE_N else rng.choice(x, size=_NORM_SAMPLE_N,
                                                          replace=False)
        # Hist clipped at p99 so the tail doesn't flatten the bulk.
        p99 = np.quantile(plot_x, 0.99)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"Normality diagnostics - {var}  ({label})", fontsize=12)
        sns.histplot(plot_x[plot_x <= p99], bins=60, kde=True, ax=axes[0],
                     color="#4c72b0", edgecolor="white", linewidth=0.3)
        axes[0].set_title(f"Distribution (skew={skew:+.2f}, kurt={kurt:+.2f})")
        axes[0].set_xlabel(var)

        sns.boxplot(x=plot_x, ax=axes[1], color="#dd8452")
        axes[1].set_title("Boxplot")
        axes[1].set_xlabel(var)

        stats.probplot(plot_x, dist="norm", plot=axes[2])
        axes[2].set_title("Q-Q normal")

        _save(fig, f"ext_normality_{label.lower()}_{var.lower()}")

    out = pd.DataFrame(rows)
    out_path = cfg.EDA_DIR / f"ext_normality_{label.lower()}.csv"
    out.to_csv(out_path, index=False)
    print(f"  [table] {out_path}")
    if len(out):
        print(out[["variable", "n", "skew", "kurt_excess",
                   "shapiro_p", "dagostino_p", "anderson_p_approx",
                   "normal_at_5pct"]].to_string(index=False))
    return out


def _eta_squared(ss_effect: float, ss_total: float) -> float:
    if ss_total <= 0:
        return float("nan")
    return float(ss_effect / ss_total)


def _omega_squared(ss_effect: float, df_effect: int,
                   ms_error: float, ss_total: float) -> float:
    if ss_total <= 0:
        return float("nan")
    return float((ss_effect - df_effect * ms_error) / (ss_total + ms_error))


def anova_one_way(df_op: pd.DataFrame, label: str) -> dict:
    """One-way ANOVA of log_Precio on Inmobiliaria for one operation subsample.
    """
    if "log_Precio" not in df_op.columns or "Inmobiliaria" not in df_op.columns:
        return {}

    print(f"\n{'-'*60}\n  ONE-WAY ANOVA  log_Precio ~ Inmobiliaria  -  {label}\n{'-'*60}")
    g1 = pd.to_numeric(df_op.loc[df_op["Inmobiliaria"] == 1, "log_Precio"],
                       errors="coerce").dropna().to_numpy()
    g0 = pd.to_numeric(df_op.loc[df_op["Inmobiliaria"] == 0, "log_Precio"],
                       errors="coerce").dropna().to_numpy()
    if len(g1) < 10 or len(g0) < 10:
        print("  Not enough data in one of the groups; skipped.")
        return {}

    # Levene
    lev_stat, lev_p = stats.levene(g1, g0, center="median")

    # Classical ANOVA (equal-variance F)
    f_stat, f_p = stats.f_oneway(g1, g0)

    # Welch's ANOVA (unequal variances)
    t_w, p_w = stats.ttest_ind(g1, g0, equal_var=False)

    # Kruskal-Wallis
    kw_stat, kw_p = stats.kruskal(g1, g0)

    # Effect sizes
    n1, n0 = len(g1), len(g0)
    m1, m0 = g1.mean(), g0.mean()
    pooled_sd = np.sqrt(((n1 - 1) * g1.var(ddof=1)
                         + (n0 - 1) * g0.var(ddof=1)) / (n1 + n0 - 2))
    cohens_d = (m1 - m0) / pooled_sd if pooled_sd > 0 else np.nan

    # eta^2 and omega^2 from the classical decomposition
    all_x  = np.concatenate([g1, g0])
    ss_t   = np.sum((all_x - all_x.mean()) ** 2)
    ss_b   = (n1 * (m1 - all_x.mean()) ** 2
              + n0 * (m0 - all_x.mean()) ** 2)
    ss_w   = ss_t - ss_b
    df_b   = 1
    df_w   = n1 + n0 - 2
    ms_w   = ss_w / df_w if df_w > 0 else np.nan
    eta2   = _eta_squared(ss_b, ss_t)
    om2    = _omega_squared(ss_b, df_b, ms_w, ss_t)

    out = {
        "subsample": label,
        "n_agency": n1,
        "n_private": n0,
        "mean_log_agency": float(m1),
        "mean_log_private": float(m0),
        "diff_log_means": float(m1 - m0),
        "exp_diff_pct": float((np.exp(m1 - m0) - 1) * 100),
        "levene_stat": float(lev_stat),
        "levene_p": float(lev_p),
        "F_classical": float(f_stat),
        "F_p": float(f_p),
        "Welch_t": float(t_w),
        "Welch_p": float(p_w),
        "KW_stat": float(kw_stat),
        "KW_p": float(kw_p),
        "cohens_d": float(cohens_d),
        "eta_squared": eta2,
        "omega_squared": om2,
    }
    print(f"  N agency / private:         {n1:,} / {n0:,}")
    print(f"  Mean log_Precio (agency):   {m1:+.4f}")
    print(f"  Mean log_Precio (private):  {m0:+.4f}")
    print(f"  exp(diff) - 1:              {out['exp_diff_pct']:+.2f}%")
    print(f"  Levene  p = {lev_p:.4g}   (variances equal? {'yes' if lev_p > 0.05 else 'NO'})")
    print(f"  ANOVA   F = {f_stat:,.2f}  p = {f_p:.4g}")
    print(f"  Welch   t = {t_w:+.2f}   p = {p_w:.4g}")
    print(f"  KW      H = {kw_stat:,.2f}  p = {kw_p:.4g}")
    print(f"  Cohen's d = {cohens_d:+.4f}   eta^2 = {eta2:.4f}   omega^2 = {om2:.4f}")
    return out


def anova_two_way(df_op: pd.DataFrame, label: str) -> pd.DataFrame:
    """Two-way ANOVA: log_Precio ~ Inmobiliaria * Caracteristicas.

    Tests the agency main effect, the property-type main effect, and their
    interaction.
    """
    if not {"log_Precio", "Inmobiliaria", "Caracteristicas"}.issubset(df_op.columns):
        return pd.DataFrame()

    print(f"\n{'-'*60}\n  TWO-WAY ANOVA  log_Precio ~ Inm * Tipo  -  {label}\n{'-'*60}")
    sub = df_op[["log_Precio", "Inmobiliaria", "Caracteristicas"]].dropna().copy()

    # Drop tiny property-type levels
    counts = sub["Caracteristicas"].value_counts()
    keep = counts[counts >= cfg.MIN_N_PER_HET_BUCKET].index
    sub = sub[sub["Caracteristicas"].isin(keep)].copy()
    if sub.empty or sub["Caracteristicas"].nunique() < 2:
        print(f"  Not enough levels with >= {cfg.MIN_N_PER_HET_BUCKET} obs; skipping.")
        return pd.DataFrame()

    print(f"  Levels retained (>= {cfg.MIN_N_PER_HET_BUCKET} obs): "
          f"{sub['Caracteristicas'].nunique()}  "
          f"({', '.join(sorted(keep.astype(str)))})")
    print(f"  Observations: {len(sub):,}")

    model = smf.ols(
        "log_Precio ~ C(Inmobiliaria) * C(Caracteristicas)",
        data=sub,
    ).fit()
    table = anova_lm(model, typ=2)
    table = table.rename(columns={"sum_sq": "SS", "df": "df",
                                  "F": "F", "PR(>F)": "p"})

    # Partial eta^2 = SS_effect / (SS_effect + SS_residual)
    ss_res = float(table.loc["Residual", "SS"])
    table["partial_eta2"] = table["SS"] / (table["SS"] + ss_res)
    table.loc["Residual", "partial_eta2"] = np.nan

    out_path = cfg.EDA_DIR / f"ext_anova2_{label.lower()}.csv"
    table.round(6).to_csv(out_path)
    print(f"  [table] {out_path}")
    print(table.round(4).to_string())

    means = (sub.groupby(["Caracteristicas", "Inmobiliaria"])["log_Precio"]
                .mean().unstack("Inmobiliaria"))
    means = means.assign(_o=means.mean(axis=1)).sort_values("_o").drop(columns="_o")

    fig, ax = plt.subplots(figsize=(max(7, 0.6 * len(means) + 4), 5))
    x = np.arange(len(means))
    if 0 in means.columns:
        ax.plot(x, means[0].values, marker="o", linewidth=2,
                color=cfg.COLORES_INM["Particular"], label="Private")
    if 1 in means.columns:
        ax.plot(x, means[1].values, marker="s", linewidth=2,
                color=cfg.COLORES_INM["Inmobiliaria"], label="Agency")
    ax.set_xticks(x)
    ax.set_xticklabels(means.index, rotation=30, ha="right")
    ax.set_ylabel("Mean log(Precio)")
    ax.set_title(f"Interaction: channel x property type - {label}")
    ax.legend()
    _save(fig, f"ext_anova2_interaction_{label.lower()}")
    return table


def bivariate_plots(df_op: pd.DataFrame, label: str) -> None:
    """Scatter (Precio~Metros), ECDF, and violin of log_Precio by channel."""
    rng = _rng()

    # Scatter on a subsample 
    sub_cols = _available(df_op, ["Precio", "Metros", "Inmobiliaria",
                                  "Tipo_Anunciante"])
    sub = df_op[sub_cols].dropna()
    if {"Precio", "Metros", "Tipo_Anunciante"}.issubset(sub.columns) and len(sub) > 50:
        x_hi = sub["Metros"].quantile(0.99)
        y_hi = sub["Precio"].quantile(0.99)
        sub_v = sub[(sub["Metros"] <= x_hi) & (sub["Precio"] <= y_hi)]
        n = min(len(sub_v), _SCATTER_SAMPLE_N)
        idx = rng.choice(len(sub_v), size=n, replace=False)
        sample = sub_v.iloc[idx]

        fig, ax = plt.subplots(figsize=(9, 6))
        sns.scatterplot(
            data=sample, x="Metros", y="Precio",
            hue="Tipo_Anunciante",
            hue_order=["Inmobiliaria", "Particular"],
            palette=cfg.COLORES_INM,
            s=10, alpha=0.35, edgecolor="none", ax=ax,
        )
        ax.set_title(f"Price vs surface area by channel - {label} (p99 trim, n={n:,})")
        ax.set_xlabel("Surface (m^2)"); ax.set_ylabel("EUR")
        ax.yaxis.set_major_formatter(_PCT_FMT)
        ax.legend(title="")
        _save(fig, f"ext_scatter_price_metros_{label.lower()}")

    # ECDF of log_Precio by channel
    if {"log_Precio", "Tipo_Anunciante"}.issubset(df_op.columns):
        fig, ax = plt.subplots(figsize=(9, 5))
        for tipo, color in cfg.COLORES_INM.items():
            data = (df_op.loc[df_op["Tipo_Anunciante"] == tipo, "log_Precio"]
                          .dropna().to_numpy())
            if data.size < 10:
                continue
            xs = np.sort(data)
            ys = np.arange(1, xs.size + 1) / xs.size
            ax.plot(xs, ys, label=tipo, color=color, linewidth=1.8)
        ax.set_xlabel("log(Precio)"); ax.set_ylabel("F(x)")
        ax.set_title(f"Empirical CDF of log-price by channel - {label}")
        ax.legend()
        _save(fig, f"ext_ecdf_logprice_{label.lower()}")

    # Violin of log_Precio by channel
    if {"log_Precio", "Tipo_Anunciante"}.issubset(df_op.columns):
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.violinplot(
            data=df_op, x="Tipo_Anunciante", y="log_Precio",
            order=["Inmobiliaria", "Particular"],
            palette=cfg.COLORES_INM, inner="quartile", cut=0, ax=ax,
        )
        ax.set_title(f"log-price distribution by channel - {label}")
        ax.set_xlabel(""); ax.set_ylabel("log(Precio)")
        _save(fig, f"ext_violin_logprice_{label.lower()}")


def _cramers_v(table: np.ndarray) -> float:
    """Bias-corrected Cramer's V"""
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    if n == 0:
        return float("nan")
    r, c = table.shape
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (r - 1) * (c - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    c_corr = c - (c - 1) ** 2 / (n - 1)
    denom = min(r_corr - 1, c_corr - 1)
    if denom <= 0:
        return float("nan")
    return float(np.sqrt(phi2_corr / denom))


def categorical_association(df_op: pd.DataFrame, label: str) -> pd.DataFrame:
    """Chi-square + Cramer's V between Inmobiliaria and each categorical/dummy.
    """
    if "Inmobiliaria" not in df_op.columns:
        return pd.DataFrame()

    print(f"\n{'-'*60}\n  CATEGORICAL ASSOCIATION (vs Inmobiliaria) - {label}\n{'-'*60}")
    rows = []
    for var in ["Caracteristicas", "Terraza", "Piscina", "Garaje"]:
        if var not in df_op.columns:
            continue
        cross = pd.crosstab(df_op[var], df_op["Inmobiliaria"])
        if cross.shape[0] < 2 or cross.shape[1] < 2:
            continue
        chi2, p, dof, exp = stats.chi2_contingency(cross.values)
        v = _cramers_v(cross.values)
        rows.append({
            "variable": var,
            "levels": cross.shape[0],
            "n": int(cross.values.sum()),
            "chi2": float(chi2),
            "df": int(dof),
            "p": float(p),
            "cramers_v": v,
            "strength": ("negligible" if v < 0.1
                         else "weak" if v < 0.2
                         else "moderate" if v < 0.4
                         else "strong"),
        })
        if var in {"Terraza", "Piscina", "Garaje"}:
            rate = (df_op.groupby("Inmobiliaria")[var].mean() * 100).round(2)
            print(f"  {var:<14s} share with feature  agency={rate.get(1, np.nan):>5}%  "
                  f"private={rate.get(0, np.nan):>5}%   "
                  f"chi2={chi2:,.1f}  p={p:.3g}  V={v:.3f}")
        else:
            print(f"  {var:<14s} levels={cross.shape[0]:<3}  "
                  f"chi2={chi2:,.1f}  p={p:.3g}  V={v:.3f}")

    out = pd.DataFrame(rows)
    out_path = cfg.EDA_DIR / f"ext_assoc_{label.lower()}.csv"
    out.to_csv(out_path, index=False)
    print(f"  [table] {out_path}")

    if not out.empty:
        fig, ax = plt.subplots(figsize=(7, 0.55 * len(out) + 1.5))
        sns.barplot(data=out.sort_values("cramers_v"),
                    x="cramers_v", y="variable", color="#4c72b0", ax=ax)
        ax.set_xlabel("Cramer's V (bias-corrected)"); ax.set_ylabel("")
        ax.set_title(f"Association with Inmobiliaria - {label}")
        ax.set_xlim(0, max(0.3, out["cramers_v"].max() * 1.15))
        _save(fig, f"ext_assoc_cramersv_{label.lower()}")
    return out


def time_series_view(df_op: pd.DataFrame, label: str) -> None:
    """Median price by YearQuarter, plus agency share per quarter."""
    if "YearQuarter" not in df_op.columns:
        return
    sub = df_op.dropna(subset=["YearQuarter", "Precio"]).copy()
    if sub.empty:
        return
    sub["YearQuarter"] = sub["YearQuarter"].astype(str)
    sub = sub.sort_values("YearQuarter")

    by_q = sub.groupby("YearQuarter").agg(
        median_price=("Precio", "median"),
        n=("Precio", "size"),
        agency_share=("Inmobiliaria", "mean"),
    )
    by_q["agency_share_pct"] = (by_q["agency_share"] * 100).round(2)
    by_q.to_csv(cfg.EDA_DIR / f"ext_timeseries_{label.lower()}.csv")

    # Median by quarter split by channel
    by_qc = (sub.groupby(["YearQuarter", "Tipo_Anunciante"])["Precio"]
                .median().unstack("Tipo_Anunciante"))

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    if "Inmobiliaria" in by_qc.columns:
        axes[0].plot(by_qc.index, by_qc["Inmobiliaria"], marker="o",
                     color=cfg.COLORES_INM["Inmobiliaria"], label="Agency")
    if "Particular" in by_qc.columns:
        axes[0].plot(by_qc.index, by_qc["Particular"], marker="o",
                     color=cfg.COLORES_INM["Particular"], label="Private")
    axes[0].set_title(f"Median price by quarter - {label}")
    axes[0].set_ylabel("EUR (median)")
    axes[0].yaxis.set_major_formatter(_PCT_FMT)
    axes[0].legend()

    axes[1].bar(by_q.index, by_q["agency_share_pct"],
                color=cfg.COLORES_OP.get(label, "#4c72b0"), alpha=0.85)
    axes[1].set_ylabel("Agency share (%)")
    axes[1].set_xlabel("Year-Quarter")
    axes[1].set_title(f"Agency listings as share of all listings - {label}")
    for tick in axes[1].get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")
    _save(fig, f"ext_timeseries_{label.lower()}")


def run_extended_eda(df: pd.DataFrame) -> dict:
    """Run all of the new diagnostics."""
    summary = {"anova_one_way": [], "two_way": {}, "association": {},
               "normality": {}}

    # Whole-panel diagnostics
    missing_data_report(df)

    for label in ("Compra", "Alquiler"):
        df_op = df[df["Operacion"] == label].copy()
        if df_op.empty:
            print(f"[ext-eda] No rows for operation = {label}; skipping.")
            continue
        correlation_matrix(df_op, label)
        summary["normality"][label]    = normality_analysis(df_op, label)
        ow = anova_one_way(df_op, label)
        if ow:
            summary["anova_one_way"].append(ow)
        summary["two_way"][label]      = anova_two_way(df_op, label)
        bivariate_plots(df_op, label)
        summary["association"][label]  = categorical_association(df_op, label)
        time_series_view(df_op, label)

    if summary["anova_one_way"]:
        pd.DataFrame(summary["anova_one_way"]).to_csv(
            cfg.EDA_DIR / "ext_anova_one_way.csv", index=False
        )
    return summary


def run_full_eda(df: pd.DataFrame) -> None:
    """Run every descriptive routine on both operation types."""
    summaries = {}
    for label in ("Compra", "Alquiler"):
        df_op = df[df["Operacion"] == label].copy()
        if df_op.empty:
            print(f"[eda] No rows found for operation = {label}; skipping.")
            continue
        summaries[label] = describe_operation(df_op, label)
        describe_municipalities(df_op, label)

    pd.DataFrame(summaries).T.to_csv(cfg.EDA_DIR / "summary_unconditional.csv")

    # Extended diagnostics (correlations, normality, ANOVA, etc.)
    run_extended_eda(df)