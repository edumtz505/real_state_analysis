import pandas as pd
from sklearn.preprocessing import LabelEncoder

import config as cfg


def trim_outliers(
    df: pd.DataFrame,
    cols=("Precio", "Metros"),
    lower: float = cfg.OUTLIER_LOWER_Q,
    upper: float = cfg.OUTLIER_UPPER_Q,
) -> pd.DataFrame:
    """Keep rows within [lower, upper] quantiles for every column in ``cols``.
    """
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        q_lo = out[col].quantile(lower)
        q_hi = out[col].quantile(upper)
        out = out[(out[col] >= q_lo) & (out[col] <= q_hi)]
    return out


def fit_target_encoder(
    df: pd.DataFrame,
    group_col: str,
    target: str,
    smoothing: float = 0.0,
) -> tuple[dict, float]:
    """Compute the mean of ``target`` for each level of ``group_col``.
    """
    global_mean = float(df[target].mean())
    if smoothing > 0:
        agg = df.groupby(group_col)[target].agg(["mean", "count"])
        smoothed = (agg["count"] * agg["mean"] + smoothing * global_mean) / (
            agg["count"] + smoothing
        )
        encoder = smoothed.to_dict()
    else:
        encoder = df.groupby(group_col)[target].mean().to_dict()
    return encoder, global_mean

def apply_target_encoder(
    df: pd.DataFrame,
    group_col: str,
    encoder: dict,
    global_mean: float,
) -> pd.Series:
    """Map ``group_col`` through ``encoder``; unseen levels get global_mean."""
    return df[group_col].map(encoder).fillna(global_mean).astype(float)


def build_design(
    df: pd.DataFrame,
    *,
    target: str = cfg.TARGET_LOG,
    numeric: list = None,
    dummies: list = None,
    categorical: list = None,
    treatment: str = cfg.TREATMENT,
    fe_var: str = cfg.FE_VAR,
    province_var: str = cfg.PROVINCE_VAR,
    postal_var: str = cfg.POSTAL_VAR,
    time_fe_var: str = cfg.TIME_FE_VAR,
    encode_fe: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Returns all variables ready for the pipeline.
    """
    numeric = numeric or (cfg.NUMERIC_FEATURES + cfg.ENGINEERED_NUMERIC)
    dummies = dummies or cfg.DUMMY_FEATURES
    categorical = categorical or cfg.CATEGORICAL_FEATURES

    cols = (numeric + dummies + categorical
            + [treatment, fe_var, province_var, postal_var, time_fe_var, target])
    sub  = df[[c for c in cols if c in df.columns]].dropna().copy()

    if encode_fe:
        # One-hot the (low-cardinality) property-type categorical
        cat_present = [c for c in categorical if c in sub.columns]
        if cat_present:
            sub = pd.get_dummies(
                sub, columns=cat_present,
                prefix=[f"is_{c}" for c in cat_present],
                drop_first=False, dtype=int,
            )
        # Time FE: low cardinality, integer-encoded
        if time_fe_var in sub.columns:
            le_t = LabelEncoder()
            sub["YQ_enc"] = le_t.fit_transform(sub[time_fe_var].astype(str))

        # Build the feature column list
        cat_dummy_cols = [c for c in sub.columns
                          if any(c.startswith(f"is_{cat}_") for cat in cat_present)]
        feature_cols = numeric + dummies + cat_dummy_cols + [treatment]
        if fe_var in sub.columns:
            feature_cols.append(fe_var)
        if province_var in sub.columns:
            feature_cols.append(province_var)
        if postal_var in sub.columns:
            feature_cols.append(postal_var)
        if time_fe_var in sub.columns:
            feature_cols.append("YQ_enc")
    else:
        feature_cols = numeric + dummies + [treatment]

    X = sub[feature_cols].copy()
    y = sub[target]
    return X, y, sub

def filter_thin_municipalities(
    df: pd.DataFrame,
    min_listings: int = cfg.MIN_LISTINGS_MUN,
    fe_var: str = cfg.FE_VAR,
) -> pd.DataFrame:
    """Drop listings whose municipality has fewer than ``min_listings`` obs.
    """
    counts  = df[fe_var].value_counts()
    keepers = counts[counts >= min_listings].index
    return df[df[fe_var].isin(keepers)].copy()