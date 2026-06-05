import numpy as np
import pandas as pd
import config as cfg

 
_EUROPEAN_NUMERIC_COLS = [
    "Precio", "Metros", "Relacion", "Habitaciones", "Aseos",
    "Latitud", "Longitud", "Terraza", "Piscina", "Garaje",
]
 
 
def _to_numeric_european(series: pd.Series) -> pd.Series:
    """Convert European-formatted numbers (1.234.567,89) into floats.
    """
    if series.dtype != object:
        return series
    return (
        series.astype(str)
              .str.replace(".", "", regex=False)
              .str.replace(",", ".", regex=False)
              .pipe(pd.to_numeric, errors="coerce")
    )

 
_OPERATION_RE = r"fotocasa\.es/es/([^/]+)/"
_AGENCY_RE = r"/es/inmobiliaria-"
_CLIENT_ID_RE = r"clientId=(\d{13})"
 
 
def _extract_operation(urls: pd.Series) -> pd.Series:
    """Map the URL slug to the canonical operation label.
    """
    return (
        urls.astype(str)
            .str.extract(_OPERATION_RE, expand=False)
            .map({"comprar": "Compra", "alquiler": "Alquiler"})
    )
 
 
def _extract_intermediation(client_urls: pd.Series) -> pd.Series:
    """1 if the client URL marks an agency listing, 0 otherwise."""
    return client_urls.astype(str).str.contains(
        _AGENCY_RE, regex=True, na=False
    ).astype(int)
 
 
def _extract_client_id(client_urls: pd.Series) -> pd.Series:
    """Pull the 13-digit clientId where present (used for sanity checks)."""
    return client_urls.astype(str).str.extract(_CLIENT_ID_RE, expand=False)

 
def load_raw(path=None) -> pd.DataFrame:
    """Read the raw CSV with the conventions used by the dataset.
    """
    path = path or cfg.RAW_CSV
    df = pd.read_csv(
        path,
        sep=cfg.CSV_SEP,
        encoding=cfg.CSV_ENCODING,
        decimal=cfg.CSV_DECIMAL,
        thousands=cfg.CSV_THOUSANDS,
        low_memory=False,
    )
    return df
 
 
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning and feature-engineering steps on a raw dataframe.
    """
    df = df.copy()
 
    # Numeric coercion
    for col in _EUROPEAN_NUMERIC_COLS:
        if col in df.columns:
            df[col] = _to_numeric_european(df[col])
 
    # Date parsing
    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")
        mask = df["Fecha"].notna()
        df["YearQuarter"] = np.nan
        df.loc[mask, "YearQuarter"] = (
            df.loc[mask, "Fecha"].dt.year.astype(str)
            + "Q"
            + df.loc[mask, "Fecha"].dt.quarter.astype(str)
        )
 
    # Operation (sale / rental)
    if "URL" in df.columns:
        df["Operacion"] = _extract_operation(df["URL"])
 
    # Intermediation flag
    if "URL_Cliente" in df.columns:
        n_before = len(df)
        df = df[df["URL_Cliente"].notna()].copy()
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"[clean] dropped {n_dropped:,} rows with missing URL_Cliente "
                  f"({n_dropped / n_before * 100:.2f}%)")
        df["Inmobiliaria"] = _extract_intermediation(df["URL_Cliente"])
        df["Tipo_Anunciante"] = df["Inmobiliaria"].map(
            {1: "Inmobiliaria", 0: "Particular"}
        )
        df["ClientID"] = _extract_client_id(df["URL_Cliente"])
 
    # Derived numeric features
    if "Precio" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df[cfg.TARGET_LOG] = np.log(df["Precio"].where(df["Precio"] > 0))
        if "Metros" in df.columns:
            df["PrecioM2"] = (df["Precio"] / df["Metros"]).replace(
                [np.inf, -np.inf], np.nan
            )

    # Engineered features used by the hedonic and tree models.
    if "Metros" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            pos = df["Metros"].where(df["Metros"] > 0)
            df["log_Metros"] = np.log(pos)
            df["Metros2"]    = df["Metros"] ** 2
    if {"Terraza", "Piscina", "Garaje"}.issubset(df.columns):
        df["n_amenities"] = (
            df["Terraza"].fillna(0)
            + df["Piscina"].fillna(0)
            + df["Garaje"].fillna(0)
        )
    if "Aseos" in df.columns and "Habitaciones" in df.columns:
        rooms = df["Habitaciones"].replace(0, np.nan)
        df["bath_per_room"] = (df["Aseos"] / rooms).replace(
            [np.inf, -np.inf], np.nan
        )

    # Property type. Keep "Unknown"
    if "Caracteristicas" in df.columns:
        df["Caracteristicas"] = (
            df["Caracteristicas"].astype(str).str.strip()
              .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
              .fillna("Unknown")
        )

    # Postal code. Normalise to a 5-digit string.
    if "CodigoPostal" in df.columns:
        cp = df["CodigoPostal"].astype(str)
        cp = cp.str.replace(r"\.0$", "", regex=True)   # drop trailing ".0"
        cp = cp.str.replace(r"\D",   "", regex=True)   # keep digits only
        cp = cp.replace("", np.nan)
        df["CodigoPostal"] = cp.where(cp.isna(), cp.str.zfill(5))

    return df
 
 
def write_clean(df: pd.DataFrame, path=None) -> None:
    """Persist the cleaned dataframe so other scripts can skip the cleaning."""
    path = path or cfg.CLEAN_CSV
    df.to_csv(path, index=False, sep=cfg.CSV_SEP, encoding="utf-8")
 
 
def load_clean(path=None) -> pd.DataFrame:
    """Read the cleaned dataframe (UTF-8, semicolon-separated)."""
    path = path or cfg.CLEAN_CSV
    df = pd.read_csv(path, sep=cfg.CSV_SEP, encoding="utf-8", low_memory=False)
    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    return df
 
 
def split_by_operation(df: pd.DataFrame) -> dict:
    """Return {'Compra': df_compra, 'Alquiler': df_alquiler}."""
    return {
        "Compra": df[df["Operacion"] == "Compra"].copy(),
        "Alquiler" : df[df["Operacion"] == "Alquiler"].copy(),
    }
 
 
def summary(df: pd.DataFrame) -> None:
    """Print a short ingest report. Useful from the CLI."""
    n = len(df)
    n_compra = int((df["Operacion"] == "Compra").sum())
    n_alquiler = int((df["Operacion"] == "Alquiler").sum())
    n_inm = int((df["Inmobiliaria"] == 1).sum())
    n_par = int((df["Inmobiliaria"] == 0).sum())
    print("=" * 60)
    print(" DATA LOAD SUMMARY")
    print("=" * 60)
    print(f"  Total listings:       {n:>10,}")
    print(f"  Sale (Compra):        {n_compra:>10,}")
    print(f"  Rental (Alquiler):    {n_alquiler:>10,}")
    print(f"  Agency listings:      {n_inm:>10,}")
    print(f"  Private listings:     {n_par:>10,}")
    if "ClientID" in df.columns:
        print(f"  Unique client IDs:    {df['ClientID'].nunique():>10,}")