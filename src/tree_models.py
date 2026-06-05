import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import shap
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from xgboost import XGBRegressor
import config as cfg
import features as ft

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def _apply_location_encoding(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    location_cols=(cfg.FE_VAR, cfg.PROVINCE_VAR, cfg.POSTAL_VAR),
    smoothing: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a smoothed target encoder per location column on train
    data and apply to both partitions, then drop the raw string columns.
    """
    train_with_target = X_train.copy()
    train_with_target["_y_"] = y_train.values

    for col in location_cols:
        if col not in X_train.columns:
            continue
        encoder, gmean = ft.fit_target_encoder(
            train_with_target, col, "_y_", smoothing=smoothing
        )
        new_col = f"{col}_te"
        X_train[new_col] = ft.apply_target_encoder(X_train, col, encoder, gmean)
        X_test[new_col] = ft.apply_target_encoder(X_test,  col, encoder, gmean)

    drop = [c for c in location_cols if c in X_train.columns]
    X_train = X_train.drop(columns=drop)
    X_test = X_test.drop(columns=drop)
    return X_train, X_test


def _tune_rf(X_train: pd.DataFrame, y_train: pd.Series, label: str) -> dict:
    """RandomizedSearchCV over a small RF grid using a reduced forest size."""
    base = {k: v for k, v in cfg.RF_PARAMS.items()
            if k not in cfg.RF_TUNE_SEARCH_SPACE}
    base["n_estimators"] = cfg.RF_CV_N_ESTIMATORS

    print(f"  Tuning RF ({label}) via RandomizedSearchCV: "
          f"{cfg.RF_TUNE_N_ITER} trials x {cfg.RF_TUNE_CV_FOLDS}-fold CV "
          f"with n_estimators={cfg.RF_CV_N_ESTIMATORS}...")
    search = RandomizedSearchCV(
        RandomForestRegressor(**base),
        param_distributions = cfg.RF_TUNE_SEARCH_SPACE,
        n_iter = cfg.RF_TUNE_N_ITER,
        cv = cfg.RF_TUNE_CV_FOLDS,
        scoring = "r2",
        random_state = cfg.SEED,
        n_jobs = 1,
        verbose = 0,
        refit = False,
    )
    search.fit(X_train, y_train)
    print(f"  Best tune CV R^2: {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")
    return {**cfg.RF_PARAMS, **search.best_params_}


def _tune_xgb(X_train: pd.DataFrame, y_train: pd.Series,
              label: str) -> tuple[dict, int]:
    """Optuna TPE search for XGB.
    Each fold runs with early stopping, and the mean best_iteration across
    folds of the winning trial is returned alongside the params
    """
    n_folds_tune = 3
    kf = KFold(n_splits=n_folds_tune, shuffle=True, random_state=cfg.SEED)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": 2000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 15),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "tree_method": "hist",
            "early_stopping_rounds": 50,
            "random_state": cfg.SEED,
            "n_jobs": -1,
            "verbosity": 0,
        }
        fold_scores, fold_best_iters = [], []
        for tr_idx, va_idx in kf.split(X_train):
            X_tr_f = X_train.iloc[tr_idx]; y_tr_f = y_train.iloc[tr_idx]
            X_va_f = X_train.iloc[va_idx]; y_va_f = y_train.iloc[va_idx]
            model = XGBRegressor(**params)
            model.fit(X_tr_f, y_tr_f,
                      eval_set=[(X_va_f, y_va_f)], verbose=False)
            fold_scores.append(
                float(r2_score(y_va_f, model.predict(X_va_f)))
            )
            # best_iteration is set by early stopping; fall back to n_estimators
            # if it didn't trigger.
            fold_best_iters.append(
                int(getattr(model, "best_iteration", params["n_estimators"]))
            )
        trial.set_user_attr(
            "best_iter_mean", int(np.ceil(np.mean(fold_best_iters)))
        )
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=cfg.SEED),
    )

    # Warm-start: queue cfg.XGB_PARAMS as trial 0. If no later trial beats it,
    # Optuna's best_params will *be* the defaults.
    study.enqueue_trial({
        "learning_rate": cfg.XGB_PARAMS["learning_rate"],
        "max_depth": cfg.XGB_PARAMS["max_depth"],
        "subsample": cfg.XGB_PARAMS["subsample"],
        "colsample_bytree": cfg.XGB_PARAMS["colsample_bytree"],
        "min_child_weight": cfg.XGB_PARAMS["min_child_weight"],
        "reg_alpha": cfg.XGB_PARAMS["reg_alpha"],
        "reg_lambda": cfg.XGB_PARAMS["reg_lambda"],
        "gamma": 0.0,
    })

    print(f"  Tuning XGB ({label}) via Optuna: "
          f"{cfg.XGB_TUNE_N_TRIALS} trials, TPE sampler, "
          f"{n_folds_tune}-fold CV objective, defaults warm-started.")
    study.optimize(objective, n_trials=cfg.XGB_TUNE_N_TRIALS,
                   show_progress_bar=False)

    best_n_est = int(study.best_trial.user_attrs.get(
        "best_iter_mean", cfg.XGB_CV_N_ESTIMATORS
    ))
    print(f"  Best tune CV R^2: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    print(f"  Best n_estimators (mean best_iter across folds): {best_n_est}")
    return {**cfg.XGB_PARAMS, **study.best_params}, best_n_est


def _evaluate(model, X_test, y_test, label: str, model_name: str) -> dict:
    """Compute log-scale and EUR-scale metrics on a held-out test set."""
    y_pred_log = model.predict(X_test)
    y_pred_eur = np.exp(y_pred_log)
    y_test_eur = np.exp(y_test)
 
    metrics = {
        "rmse_log": float(np.sqrt(mean_squared_error(y_test, y_pred_log))),
        "mae_log": float(mean_absolute_error(y_test, y_pred_log)),
        "r2_log": float(r2_score(y_test, y_pred_log)),
        "rmse_eur": float(np.sqrt(mean_squared_error(y_test_eur, y_pred_eur))),
        "mae_eur": float(mean_absolute_error(y_test_eur, y_pred_eur)),
    }
 
    print(f"\n  {model_name} test metrics ({label}):")
    print(f"    RMSE (log): {metrics['rmse_log']:.4f}")
    print(f"    MAE  (log): {metrics['mae_log']:.4f}")
    print(f"    R^2  (log): {metrics['r2_log']:.4f}")
    print(f"    RMSE (EUR): {metrics['rmse_eur']:>12,.0f}")
    print(f"    MAE  (EUR): {metrics['mae_eur']:>12,.0f}")
    return metrics

 
def _plot_feature_importance(
    feature_names: list,
    importances: np.ndarray,
    label: str,
    model_name: str,
    *,
    xlabel: str,
) -> None:
    order = np.argsort(importances)[::-1]
    feats = [feature_names[i] for i in order]
    vals = importances[order]
    colors = ["tomato" if f == cfg.TREATMENT else "steelblue" for f in feats]

    height = max(5, 0.3 * len(feats) + 1)
    fig, ax = plt.subplots(figsize=(9, height))
    ax.barh(feats[::-1], vals[::-1], color=colors[::-1], alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_title(f"{model_name} - feature importance ({label})")
    rank = feats.index(cfg.TREATMENT) + 1
    ax.text(0.98, 0.02,
            f"{cfg.TREATMENT}: {vals[rank - 1]:.4f} (rank {rank}/{len(feats)})",
            transform=ax.transAxes, ha="right", va="bottom",
            color="tomato",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="tomato", alpha=0.9))
    fig.tight_layout()
    out = cfg.FIG_DIR / f"{model_name.lower()}_importance_{label.lower()}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {out}")
 
 
def _plot_shap_summary(shap_values, X_sample, label, model_name) -> None:
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_sample, show=False, plot_size=None)
    plt.title(f"{model_name} - SHAP summary ({label})")
    out = cfg.FIG_DIR / f"{model_name.lower()}_shap_summary_{label.lower()}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [figure] {out}")
 
 
def _plot_shap_treatment(shap_values, X_sample, label, model_name) -> None:
    """Distribution of SHAP for the Inmobiliaria column, by group."""
    if cfg.TREATMENT not in X_sample.columns:
        return
    col_idx = list(X_sample.columns).index(cfg.TREATMENT)
    shap_t = shap_values[:, col_idx]
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{model_name} - SHAP for {cfg.TREATMENT} ({label})")
 
    is_agency = X_sample[cfg.TREATMENT].values == 1
    is_private = X_sample[cfg.TREATMENT].values == 0
 
    axes[0].hist(shap_t[is_agency],  bins=50, alpha=0.7,
                 color="steelblue", label="Agency",  density=True)
    axes[0].hist(shap_t[is_private], bins=50, alpha=0.7,
                 color="tomato",    label="Private", density=True)
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("SHAP value (impact on log-price)")
    axes[0].set_ylabel("Density"); axes[0].legend()
    axes[0].set_title("Distribution by channel")
 
    axes[1].scatter(X_sample[cfg.TREATMENT].values, shap_t,
                    alpha=0.05, s=3,
                    c=["steelblue" if v == 1 else "tomato"
                       for v in X_sample[cfg.TREATMENT].values])
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["Private", "Agency"])
    axes[1].set_xlabel("Channel"); axes[1].set_ylabel("SHAP value")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("SHAP value vs channel")
 
    fig.tight_layout()
    out = cfg.FIG_DIR / f"{model_name.lower()}_shap_treatment_{label.lower()}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {out}")


def fit_random_forest(df_op: pd.DataFrame, label: str) -> dict:
    """Train an RF on the operation subsample, report metrics and SHAP."""
    print(f" RANDOM FOREST ")
    sub = ft.filter_thin_municipalities(df_op)
    X, y, _ = ft.build_design(sub, encode_fe=True)
    print(f"  Observations: {len(X):,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.TEST_SIZE, random_state=cfg.SEED,
        stratify=X[cfg.TREATMENT],
    )

    # Trim outliers
    metros_lo = X_train["Metros"].quantile(cfg.OUTLIER_LOWER_Q)
    metros_hi = X_train["Metros"].quantile(cfg.OUTLIER_UPPER_Q)
    precio_lo = np.exp(y_train).quantile(cfg.OUTLIER_LOWER_Q)
    precio_hi = np.exp(y_train).quantile(cfg.OUTLIER_UPPER_Q)
    train_mask = (
        (X_train["Metros"] >= metros_lo) & (X_train["Metros"] <= metros_hi) &
        (np.exp(y_train) >= precio_lo)   & (np.exp(y_train) <= precio_hi)
    )
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]

    # Leak-safe target encoding for high-cardinality location columns
    X_train, X_test = _apply_location_encoding(X_train, X_test, y_train)
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}  |  Features: {X_train.shape[1]}")

    rf_params = _tune_rf(X_train, y_train, label)

    # CV uses a smaller forest than the final model (RF_CV_N_ESTIMATORS) to
    # bound memory.
    print(f"  Running {cfg.N_FOLDS_CV}-fold CV on training set...")
    kf = KFold(n_splits=cfg.N_FOLDS_CV, shuffle=True, random_state=cfg.SEED)
    cv_params = {**rf_params, "n_estimators": cfg.RF_CV_N_ESTIMATORS}
    cv_r2 = cross_val_score(
        RandomForestRegressor(**cv_params),
        X_train, y_train, cv=kf, scoring="r2", n_jobs=1,
    )
    print(f"  CV R^2: {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}  "
          f"(n_estimators={cfg.RF_CV_N_ESTIMATORS} during CV)")

    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_train, y_train)

    metrics = _evaluate(rf, X_test, y_test, label, "RandomForest")
    metrics["cv_r2_mean"] = float(cv_r2.mean())
    metrics["cv_r2_std"] = float(cv_r2.std())

    _plot_feature_importance(
        list(X_train.columns), rf.feature_importances_, label,
        "RandomForest", xlabel="Mean Decrease in Impurity",
    )

    print(f"  Computing SHAP on {cfg.SHAP_SAMPLE_SIZE} test rows...")
    rng = np.random.RandomState(cfg.SEED)
    sample_ix = rng.choice(len(X_test),
                           size=min(cfg.SHAP_SAMPLE_SIZE, len(X_test)),
                           replace=False)
    X_sample = X_test.iloc[sample_ix].reset_index(drop=True)
    shap_vals = shap.TreeExplainer(rf).shap_values(X_sample)
    _plot_shap_summary(shap_vals, X_sample, label, "RandomForest")
    _plot_shap_treatment(shap_vals, X_sample, label, "RandomForest")

    shap_imp = pd.DataFrame({
        "Variable": list(X_train.columns),
        "SHAP_mean_abs": np.abs(shap_vals).mean(axis=0),
    }).sort_values("SHAP_mean_abs", ascending=False)
    shap_imp.to_csv(
        cfg.MODEL_DIR / f"rf_shap_importance_{label.lower()}.csv", index=False
    )
    print("\n  Mean SHAP ranking:")
    print(shap_imp.to_string(index=False))
    return metrics

 
def fit_xgboost(df_op: pd.DataFrame, label: str) -> dict:
    """Train XGBoost with early stopping, report metrics and SHAP."""
    print(f"\n{'-'*60}\n  XGBOOST - {label}\n{'-'*60}")
    sub = ft.filter_thin_municipalities(df_op)
    X, y, _ = ft.build_design(sub, encode_fe=True)
    print(f"  Observations: {len(X):,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.TEST_SIZE, random_state=cfg.SEED,
        stratify=X[cfg.TREATMENT],
    )

    # Trim outliers
    metros_lo = X_train["Metros"].quantile(cfg.OUTLIER_LOWER_Q)
    metros_hi = X_train["Metros"].quantile(cfg.OUTLIER_UPPER_Q)
    precio_lo = np.exp(y_train).quantile(cfg.OUTLIER_LOWER_Q)
    precio_hi = np.exp(y_train).quantile(cfg.OUTLIER_UPPER_Q)
    train_mask = (
        (X_train["Metros"] >= metros_lo) & (X_train["Metros"] <= metros_hi) &
        (np.exp(y_train) >= precio_lo) & (np.exp(y_train) <= precio_hi)
    )
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]

    # Leak-safe target encoding for high-cardinality location columns.
    X_train, X_test = _apply_location_encoding(X_train, X_test, y_train)

    xgb_params, tuned_n_est = _tune_xgb(X_train, y_train, label)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.10, random_state=cfg.SEED
    )
    print(f"  Train: {len(X_tr):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}  |  Features: {X_train.shape[1]}")

    print(f"  Running {cfg.N_FOLDS_CV}-fold CV on training set "
          f"(n_estimators={tuned_n_est})...")
    cv_params = {k: v for k, v in xgb_params.items()
                 if k != "early_stopping_rounds"}
    cv_params["n_estimators"] = tuned_n_est
    kf = KFold(n_splits=cfg.N_FOLDS_CV, shuffle=True, random_state=cfg.SEED)
    cv_r2 = cross_val_score(
        XGBRegressor(**cv_params),
        X_train, y_train, cv=kf, scoring="r2", n_jobs=-1,
    )
    print(f"  CV R^2: {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")

    xgb = XGBRegressor(**xgb_params)
    xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        verbose=False,
    )
    if hasattr(xgb, "best_iteration"):
        print(f"  Best iteration: {xgb.best_iteration}")

    metrics = _evaluate(xgb, X_test, y_test, label, "XGBoost")
    metrics["cv_r2_mean"] = float(cv_r2.mean())
    metrics["cv_r2_std"] = float(cv_r2.std())
    if hasattr(xgb, "best_iteration"):
        metrics["best_iteration"] = int(xgb.best_iteration)

    _plot_feature_importance(
        list(X_train.columns), xgb.feature_importances_, label,
        "XGBoost", xlabel="Gain (mean per split)",
    )
 
    # Learning curve
    try:
        results = xgb.evals_result()
        if results:
            train_rmse = results["validation_0"]["rmse"]
            val_rmse = results["validation_1"]["rmse"]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(train_rmse, label="Train RMSE", color="darkorange")
            ax.plot(val_rmse, label="Val RMSE", color="steelblue")
            ax.set_xlabel("Iteration"); ax.set_ylabel("RMSE (log scale)")
            ax.set_title(f"XGBoost learning curve - {label}")
            ax.legend(); fig.tight_layout()
            out = cfg.FIG_DIR / f"xgboost_learning_{label.lower()}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  [figure] {out}")
    except Exception as e:
        print(f"  [warn] learning-curve plot failed: {e}")
 
    print(f"  Computing SHAP on {cfg.SHAP_SAMPLE_SIZE} test rows...")
    rng       = np.random.RandomState(cfg.SEED)
    sample_ix = rng.choice(len(X_test),
                           size=min(cfg.SHAP_SAMPLE_SIZE, len(X_test)),
                           replace=False)
    X_sample  = X_test.iloc[sample_ix].reset_index(drop=True)
    shap_vals = shap.TreeExplainer(xgb).shap_values(X_sample)
    _plot_shap_summary(shap_vals, X_sample, label, "XGBoost")
    _plot_shap_treatment(shap_vals, X_sample, label, "XGBoost")
 
    shap_imp = pd.DataFrame({
        "Variable": list(X_train.columns),
        "SHAP_mean_abs": np.abs(shap_vals).mean(axis=0),
    }).sort_values("SHAP_mean_abs", ascending=False)
    shap_imp.to_csv(
        cfg.MODEL_DIR / f"xgb_shap_importance_{label.lower()}.csv", index=False
    )
    print("\n  Mean |SHAP| ranking:")
    print(shap_imp.to_string(index=False))
    return metrics

 
def run(df: pd.DataFrame, *, models=("rf", "xgb")) -> dict:
    """Run RF and/or XGBoost on both operation subsamples."""
    out = {"rf": {}, "xgb": {}}
    for label in ("Compra", "Alquiler"):
        df_op = df[df["Operacion"] == label].copy()
        if df_op.empty:
            print(f"[tree_models] No rows for {label}; skipping.")
            continue
        if "rf" in models:
            out["rf"][label] = fit_random_forest(df_op, label)
        if "xgb" in models:
            out["xgb"][label] = fit_xgboost(df_op, label)
 
    # Persist consolidated metrics tables
    for model_name in ("rf", "xgb"):
        rows = [{"Submuestra": lab, **m} for lab, m in out[model_name].items()]
        if rows:
            pd.DataFrame(rows).to_csv(
                cfg.MODEL_DIR / f"{model_name}_metrics.csv", index=False
            )
    return out