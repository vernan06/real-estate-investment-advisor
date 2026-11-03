"""Train, compare, and save the Real Estate Investment Advisor models."""

import hashlib
import json
import random
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "india_housing_prices.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


def dataset_hash() -> str:
    return hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["Price_per_SqFt"] = pd.to_numeric(result["Price_per_SqFt"], errors="coerce")
    result["Age_of_Property"] = 2026 - result["Year_Built"]
    result["School_Category"] = pd.cut(
        result["Nearby_Schools"], [-1, 0, 2, 4, 6, np.inf],
        labels=["No Schools", "1-2 Schools", "3-4 Schools", "5-6 Schools", "7+ Schools"],
    ).astype(str)
    result["Hospital_Category"] = pd.cut(
        result["Nearby_Hospitals"], [-1, 0, 2, 4, 6, np.inf],
        labels=["No Hospitals", "1-2 Hospitals", "3-4 Hospitals", "5-6 Hospitals", "7+ Hospitals"],
    ).astype(str)
    result["Combined_Amenities"] = result["Nearby_Schools"] + result["Nearby_Hospitals"]
    result["Combined_Category"] = np.select(
        [result["Combined_Amenities"] <= 2, result["Combined_Amenities"] <= 5,
         result["Combined_Amenities"] <= 8],
        ["Low (0-2)", "Medium (3-5)", "High (6-8)"],
        default="Very High (9+)",
    )
    result["Parking_Space_Numeric"] = result["Parking_Space"].eq("Yes").astype(int)
    result["Price_per_SqFt_log"] = np.log1p(result["Price_per_SqFt"].clip(lower=0))
    result["Price_Range"] = pd.cut(
        result["Price_in_Lakhs"], [-np.inf, 50, 100, 200, np.inf],
        labels=["Low", "Medium", "High", "Premium"],
    ).astype(str)
    result["Furnished_Status_Num"] = result["Furnished_Status"].map(
        {"Unfurnished": 0, "Semi-furnished": 1, "Furnished": 2}
    ).fillna(0)
    result["Transport_Num"] = result["Public_Transport_Accessibility"].map(
        {"Low": 0, "Medium": 1, "High": 2}
    ).fillna(0)
    result["Investment_Score"] = (
        result["BHK"].ge(3).astype(int)
        + result["Furnished_Status_Num"].ge(1).astype(int)
        + result["Availability_Status"].ne("Under_Construction").astype(int)
    )
    return result


def classification_target(df: pd.DataFrame, median_price: float,
                          median_ppsf: float) -> np.ndarray:
    result = base_features(df)
    return (
        result["Price_in_Lakhs"].le(median_price)
        | result["Price_per_SqFt"].le(median_ppsf)
        | result["Investment_Score"].ge(2)
    ).astype(int).to_numpy()


def appreciation_target(df: pd.DataFrame) -> np.ndarray:
    # Explicit scenario: 5% base appreciation, plus feature premiums up to 3%,
    # compounded annually for five years. This is not observed market history.
    annual_rate = np.full(len(df), 0.05)
    annual_rate += np.where(df["Property_Type"].eq("Villa"), 0.01, 0.0)
    annual_rate += np.where(df["Furnished_Status"].eq("Furnished"), 0.005, 0.0)
    annual_rate += np.where(df["BHK"].ge(3), 0.005, 0.0)
    annual_rate = np.clip(annual_rate, 0.03, 0.09)
    return df["Price_in_Lakhs"].to_numpy() * np.power(1 + annual_rate, 5)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    if df.isna().any().any():
        raise ValueError("Training data contains missing values")
    if df.duplicated().any():
        raise ValueError("Training data contains duplicates")

    train_raw, test_raw = train_test_split(df, test_size=0.2, random_state=SEED, stratify=None)
    freq_maps = {
        name: train_raw[name].value_counts(normalize=True).to_dict()
        for name in ("City", "Locality")
    }
    winsor_bounds = tuple(df["Price_per_SqFt"].quantile([0.01, 0.99]))
    category_columns = ["State", "Property_Type", "Parking_Space", "Security",
                        "Facing", "Owner_Type", "Availability_Status"]
    train_raw = train_raw.copy()
    test_raw = test_raw.copy()
    category_values = {
        column: sorted(df[column].dropna().unique().tolist()[1:])
        for column in category_columns
    }

    def transform(raw):
        result = base_features(raw)
        result["Price_per_SqFt_winsor"] = result["Price_per_SqFt"].clip(*winsor_bounds)
        result["City_freq"] = result["City"].map(freq_maps["City"]).fillna(1e-6)
        result["Locality_freq"] = result["Locality"].map(freq_maps["Locality"]).fillna(1e-6)
        amenities = result["Amenities"].astype(str).str.extract(r"(\d+)")[0].fillna("0").astype(float)
        price_range = result["Price_Range"].map({"Low": 0, "Medium": 1, "High": 2, "Premium": 3}).fillna(0)
        school_category = result["School_Category"].map({
            "No Schools": 0, "1-2 Schools": 1, "3-4 Schools": 2, "5-6 Schools": 3, "7+ Schools": 4
        }).fillna(0)
        hospital_category = result["Hospital_Category"].map({
            "No Hospitals": 0, "1-2 Hospitals": 1, "3-4 Hospitals": 2, "5-6 Hospitals": 3, "7+ Hospitals": 4
        }).fillna(0)
        combined_category = result["Combined_Category"].map({
            "Low (0-2)": 0, "Medium (3-5)": 1, "High (6-8)": 2, "Very High (9+)": 3
        }).fillna(0)
        values = {
            "City": result["City"], "Locality": result["Locality"], "BHK": result["BHK"],
            "Size_in_SqFt": result["Size_in_SqFt"], "Price_in_Lakhs": result["Price_in_Lakhs"],
            "Price_per_SqFt": result["Price_per_SqFt"], "Year_Built": result["Year_Built"],
            "Furnished_Status": result["Furnished_Status_Num"], "Floor_No": result["Floor_No"],
            "Total_Floors": result["Total_Floors"], "Age_of_Property": result["Age_of_Property"],
            "Nearby_Schools": result["Nearby_Schools"], "Nearby_Hospitals": result["Nearby_Hospitals"],
            "Public_Transport_Accessibility": result["Transport_Num"], "Amenities": amenities,
            "Price_Range": price_range, "School_Category": school_category,
            "Hospital_Category": hospital_category, "Combined_Amenities": result["Combined_Amenities"],
            "Combined_Category": combined_category,
            "Parking_Space_Numeric": result["Parking_Space_Numeric"],
            "Price_per_SqFt_log": result["Price_per_SqFt_log"],
            "Price_per_SqFt_winsor": result["Price_per_SqFt_winsor"],
            "City_freq": result["City_freq"], "Locality_freq": result["Locality_freq"],
        }
        frame = pd.DataFrame(values, index=result.index)
        for column in category_columns:
            for value in category_values[column]:
                frame[f"{column}_{value}"] = result[column].eq(value).astype(int)
        frame["Investment_Score"] = result["Investment_Score"]
        return frame

    X_train, X_test = transform(train_raw), transform(test_raw)
    median_price = train_raw["Price_in_Lakhs"].median()
    median_ppsf = train_raw["Price_per_SqFt"].median()
    y_train = classification_target(train_raw, median_price, median_ppsf)
    y_test = classification_target(test_raw, median_price, median_ppsf)
    y_reg_train = appreciation_target(train_raw)
    y_reg_test = appreciation_target(test_raw)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=60, max_depth=16, min_samples_leaf=3,
                                                 n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08,
                                 subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                                 random_state=SEED, n_jobs=-1),
    }
    regressors = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=60, max_depth=16, min_samples_leaf=3,
                                               n_jobs=-1, random_state=SEED),
        "XGBoost": XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.08,
                                subsample=0.9, colsample_bytree=0.9, random_state=SEED, n_jobs=-1),
    }

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("real-estate-model-training")
    comparison = {"dataset_hash": dataset_hash(), "random_seed": SEED,
                  "classification": {}, "regression": {}, "selected": {}}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    selected_classifier = None
    selected_score = -1
    for name, estimator in classifiers.items():
        model = Pipeline([("preprocess", StandardScaler()), ("clf", estimator)])
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_test)[:, 1]
        prediction = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, prediction)),
            "precision": float(precision_score(y_test, prediction, zero_division=0)),
            "recall": float(recall_score(y_test, prediction, zero_division=0)),
            "f1": float(f1_score(y_test, prediction, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probability)),
            "cv_roc_auc_mean": float(cv_scores.mean()),
            "cv_roc_auc_std": float(cv_scores.std()),
            "confusion_matrix": confusion_matrix(y_test, prediction).tolist(),
        }
        comparison["classification"][name] = metrics
        with mlflow.start_run(run_name=f"classification-{name}"):
            mlflow.log_params({"algorithm": name, "random_seed": SEED, "dataset_hash": dataset_hash()})
            mlflow.log_metrics(metrics)
        if metrics["f1"] > selected_score:
            selected_score = metrics["f1"]
            selected_classifier = (name, model)

    selected_name, selected_model = selected_classifier
    joblib.dump(selected_model, MODEL_DIR / "gradient_boosting_model.pkl")

    selected_regressor = None
    selected_mae = float("inf")
    for name, estimator in regressors.items():
        model = Pipeline([("preprocess", StandardScaler()), ("regressor", estimator)])
        model.fit(X_train, y_reg_train)
        prediction = model.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_reg_test, prediction)),
            "rmse": float(mean_squared_error(y_reg_test, prediction) ** 0.5),
            "r2": float(r2_score(y_reg_test, prediction)),
        }
        comparison["regression"][name] = metrics
        with mlflow.start_run(run_name=f"regression-{name}"):
            mlflow.log_params({"algorithm": name, "random_seed": SEED, "dataset_hash": dataset_hash(),
                               "target_type": "assumption-based-five-year-appreciation"})
            mlflow.log_metrics(metrics)
        if metrics["mae"] < selected_mae:
            selected_mae = metrics["mae"]
            selected_regressor = (name, model)

    reg_name, reg_model = selected_regressor
    joblib.dump(reg_model, MODEL_DIR / "gradient_boosting_regressor.pkl")
    comparison["selected"] = {
        "classification": selected_name,
        "classification_artifact": "models/gradient_boosting_model.pkl",
        "regression": reg_name,
        "regression_artifact": "models/gradient_boosting_regressor.pkl",
    }
    comparison["target_definitions"] = {
        "Good_Investment": (
            "Rule-based proxy: price <= training median OR price per sqft <= training median "
            "OR Investment_Score >= 2. No observed investment outcomes exist."
        ),
        "Future_Price_5Y": (
            "Assumption-based scenario only. Base annual appreciation 5%, with a 1 percentage-point "
            "Villa premium, 0.5-point furnished premium, and 0.5-point BHK>=3 premium, capped "
            "at 3%-9% and compounded for five years. No observed resale prices exist."
        ),
    }
    comparison["feature_schema"] = list(X_train.columns)
    comparison["dataset"] = {
        "rows": len(df), "columns": len(df.columns), "missing_values": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()), "hash_sha256": dataset_hash(),
    }
    (MODEL_DIR / "training_report.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    joblib.dump({"frequency_maps": freq_maps, "winsor_bounds": winsor_bounds,
                 "median_price": median_price, "median_ppsf": median_ppsf,
                 "category_values": category_values}, MODEL_DIR / "training_metadata.joblib")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
