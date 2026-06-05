# TFM
Real Estate Intermediation in the Spanish Housing Market
Reproducible code for a study of the price premium associated with real estate
agency listings versus private-seller listings on the platform,
combining hedonic regression and tree-based machine learning.
The pipeline takes a single CSV (sales and rental listings),
extracts a binary intermediation flag from the listing URL, and estimates the
conditional price differential between agency and private listings using
four model variants: linear OLS, log-linear OLS, Random Forest and XGBoost.
Both unconditional descriptive statistics and conditional model estimates
are reported separately for the sale (Compra) and rental (Alquiler)
subsamples.

## Repository layout
project/
    data/
        DatosViviendas1.csv
    outputs/
        eda/
        models/
        figures/
    src/
        config.py
        data_loader.py
        features.py
        eda.py
        hedonic.py
        tree_models.py
        comparison.py
        heterogeneity.py
        robustness.py
        main.py 
    requirements.txt
    README.txt

## Setup
pip install -r requirements.txt
Place the raw CSV at data/. Prepare only needs to be run once. The other steps read the cleaned CSV
directly.

## Running the pipeline
python -m main all

## Or step by step
python -m main prepare         # cleaned data
python -m main eda             # descriptive analysis
python -m main hedonic         # OLS hedonic regressions
python -m main rf              # Random Forest with SHAP
python -m main xgb             # XGBoost with SHAP
python -m main compare         # cross-model comparison
python -m main heterogeneity   # interaction tests
python -m main robustness      # spec-sensitivity table for the agency coef
