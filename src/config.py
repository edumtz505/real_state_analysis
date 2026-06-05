"""
config.py
---------
Central configuration: paths, hyperparameters and modelling constants
used across the pipeline. Adjust here, not inside individual modules.
"""
 
from pathlib import Path
 
# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
EDA_DIR = OUTPUT_DIR / "eda"
MODEL_DIR = OUTPUT_DIR / "models"
FIG_DIR = OUTPUT_DIR / "figures"
 
for d in (DATA_DIR, OUTPUT_DIR, EDA_DIR, MODEL_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)
 
RAW_CSV = DATA_DIR / "DatosViviendas1.csv"
CLEAN_CSV = DATA_DIR / "fotocasa_clean.csv"

CSV_SEP = ";"
CSV_ENCODING = "latin-1"
CSV_DECIMAL  = ","
CSV_THOUSANDS = "."

SEED = 42
TEST_SIZE = 0.20
N_FOLDS_CV = 10
OUTLIER_LOWER_Q = 0.01
OUTLIER_UPPER_Q = 0.99
MIN_LISTINGS_MUN = 100
MIN_PRIVATE_MUN = 10
MIN_N_PER_HET_BUCKET = 500

NUMERIC_FEATURES = ["Metros", "Habitaciones", "Aseos"]

ENGINEERED_NUMERIC = ["log_Metros", "Metros2", "n_amenities", "bath_per_room"]

DUMMY_FEATURES = ["Terraza", "Piscina", "Garaje"]
CATEGORICAL_FEATURES = ["Caracteristicas"]   # property type (Flat, Chalet, ...)

TREATMENT = "Inmobiliaria"
FE_VAR = "NMUN"
PROVINCE_VAR = "NPRO"
POSTAL_VAR  = "CodigoPostal"
TIME_FE_VAR = "YearQuarter"
 
TARGET_LIN = "Precio"
TARGET_LOG = "log_Precio"

RF_PARAMS = {
    "n_estimators": 500,
    "max_features": "sqrt",
    "max_depth": 30,
    "min_samples_leaf": 2,
    "random_state": SEED,
    "n_jobs": -1,
}

XGB_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 15,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0,
}

XGB_CV_N_ESTIMATORS = 500

RF_CV_N_ESTIMATORS = 200

RF_TUNE_N_ITER = 20
RF_TUNE_CV_FOLDS = 3
RF_TUNE_SEARCH_SPACE = {
    "max_depth": [10, 20, 30, 40, None],
    "min_samples_leaf": [1, 2, 5, 10],
    "min_samples_split": [2, 5, 10, 20],
    "max_features": ["sqrt", "log2", 0.3, 0.5],
}

XGB_TUNE_N_TRIALS = 50
XGB_TUNE_VAL_SIZE = 0.15

SHAP_SAMPLE_SIZE = 500
 
COLORES_OP = {"Compra":      "#4c72b0", "Alquiler":   "#dd8452"}
COLORES_INM = {"Inmobiliaria": "#c44e52", "Particular": "#55a868"}