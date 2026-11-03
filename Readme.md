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

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app/app.py
```

The application entry point can also be run from the repository root with:

```bash
python -m streamlit run app.py
```

## Training and experiments

Training workflows are documented in `notebooks/experiments.ipynb`. To start local MLflow:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Reproducibility requires running the notebook's training cells and saving the resulting
artifacts. Classification targets derived by dataset rules and synthetic regression targets
must not be represented as observed real-world outcomes.

## Tests

```bash
python -m pytest -q
```

## Deployment

Deploy this repository to Streamlit Community Cloud using:

- Repository: `vernan06/real-estate-investment-advisor`
- Entry point: `streamlit_app/app.py`
- Python dependencies: `requirements.txt`

The public deployment URL will be added after a verified deployment.

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
