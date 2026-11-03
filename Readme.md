# Real Estate Investment Advisor: Predicting Property Profitability & Future Value

A Streamlit and scikit-learn application for exploring Indian residential property data,
classifying investment potential, and estimating five-year modeled property value.

## Features

- Interactive investment classification with model confidence
- Five-year price prediction from a saved regression pipeline
- Dashboard metrics and market summaries
- Twenty cross-sectional property analytics views
- Model artifact loading and evaluation documentation
- Assumption and limitation disclosures

## Data

The supplied `data/india_housing_prices.csv` contains 250,000 complete synthetic records
across 23 columns. It contains no historical observation dates, so analyses labelled as
trends are cross-sectional comparisons rather than verified historical time series.

## Setup

Install the project dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Run the application from the repository root with the root entry point:

```bash
python -m streamlit run app.py
```

The application loads the dataset and saved model artifacts using paths relative to the
application files, so it does not depend on a machine-specific working directory.

## Model training

The saved models are ready for inference; training is not needed to run the application.
To intentionally retrain and compare models, run this command from the repository root:

```bash
python train_models.py
```

The script reads `data/india_housing_prices.csv`, rejects missing or duplicate records, then
trains and compares three classifiers and three regressors using a fixed random seed. It
selects a classifier by F1 score and a regressor by mean absolute error. It overwrites the
saved inference pipelines under `models/`, writes a training report and metadata there, and
logs runs to the root `mlflow.db` SQLite database. This training workload is substantially
heavier than application startup. The classification target is a rule-based proxy, and the
five-year regression target is an assumption-based scenario, not an observed market outcome.

To inspect the local MLflow runs after training:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Tests

```bash
python -m pytest -q
```

## Deployment

For Streamlit Community Cloud, select the `main` branch and configure:

- Repository: `vernan06/real-estate-investment-advisor`
- Main file path: `app.py`
- Python dependencies: `requirements.txt` (installed automatically by the platform)

The public deployment URL will be added after deployment is verified. Review the provenance
and licensing restriction below before creating a public deployment.

## Provenance

This repository was adapted from an existing public upstream project. An upstream license
was claimed in its README, but no license file is present in the upstream Git tree and its
linked `LICENSE` file returns 404. The claim is not a substitute for the missing license
text. This repository therefore does not claim a redistribution license; obtain written
permission before further redistribution or deployment. Historical author attribution and
source provenance are documented in `ATTRIBUTION.md` without implying authorship or endorsement.

## Disclaimer

Predictions are decision-support estimates, not financial advice. The five-year forecast
is an assumption-based model output unless verified observed resale data is added.
