import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as cfg
 
def _load_hedonic() -> pd.DataFrame:
    path = cfg.MODEL_DIR / "hedonic_results_all.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
 
def _load_tree(model_name: str) -> pd.DataFrame:
    path = cfg.MODEL_DIR / f"{model_name}_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
 
def build_table() -> pd.DataFrame:
    """Build a consolidated table comparing all four model variants."""
    rows = []
 
    hed = _load_hedonic()
    for _, r in hed.iterrows():
        rows.append({
            "Submuestra": r["Submuestra"],
            "Modelo": f"OLS-{r['Especificacion']}",
            "R2_log": r["r2_adj"] if r["Especificacion"] == "log" else np.nan,
            "RMSE_log": np.nan,
            "MAE_log": np.nan,
            "RMSE_eur": np.nan,
            "MAE_eur": np.nan,
            "N": r["nobs"],
            "Coef_Inm": r["coef"],
            "p_Inm": r["pval"],
        })
 
    for model_name, label in [("rf", "RandomForest"), ("xgb", "XGBoost")]:
        df_m = _load_tree(model_name)
        for _, r in df_m.iterrows():
            rows.append({
                "Submuestra": r["Submuestra"],
                "Modelo": label,
                "R2_log": r.get("r2_log", np.nan),
                "RMSE_log": r.get("rmse_log", np.nan),
                "MAE_log": r.get("mae_log", np.nan),
                "RMSE_eur": r.get("rmse_eur", np.nan),
                "MAE_eur": r.get("mae_eur", np.nan),
                "N":  np.nan,
                "Coef_Inm": np.nan,
                "p_Inm": np.nan,
            })
 
    table = pd.DataFrame(rows)
    table.to_csv(cfg.MODEL_DIR / "model_comparison.csv", index=False)
    print(f"  [csv] {cfg.MODEL_DIR / 'model_comparison.csv'}")
    return table
 
def plot_metric_comparison() -> None:
    """Bar chart comparing log-R^2, log-RMSE and log-MAE across models."""
    rf  = _load_tree("rf")
    xgb = _load_tree("xgb")
    if rf.empty or xgb.empty:
        print("  [skip] tree-model metrics not found; run tree_models first.")
        return
 
    metrics_show = [("r2_log", "R^2 (log)"),
                    ("rmse_log", "RMSE (log)"),
                    ("mae_log", "MAE (log)")]
    submuestras = ["Compra", "Alquiler"]
 
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Test-set metrics: Random Forest vs XGBoost")
    x = np.arange(len(submuestras))
    w = 0.35
 
    for ax, (col, mlabel) in zip(axes, metrics_show):
        rf_vals = [float(rf.loc[rf["Submuestra"] == s, col].iloc[0])
                    if not rf.loc[rf["Submuestra"] == s].empty else 0
                    for s in submuestras]
        xgb_vals = [float(xgb.loc[xgb["Submuestra"] == s, col].iloc[0])
                    if not xgb.loc[xgb["Submuestra"] == s].empty else 0
                    for s in submuestras]
        b1 = ax.bar(x - w/2, rf_vals, w, label="Random Forest",
                    color="steelblue", alpha=0.85)
        b2 = ax.bar(x + w/2, xgb_vals, w, label="XGBoost",
                    color="darkorange", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(submuestras)
        ax.set_title(mlabel); ax.legend(fontsize=8)
        for b in list(b1) + list(b2):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() * 1.01,
                    f"{b.get_height():.3f}",
                    ha="center", va="bottom", fontsize=8)
 
    fig.tight_layout()
    out = cfg.FIG_DIR / "model_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {out}")
 
def run() -> None:
    print(f"\n{'-'*60}\n  MODEL COMPARISON\n{'-'*60}")
    build_table()
    plot_metric_comparison()