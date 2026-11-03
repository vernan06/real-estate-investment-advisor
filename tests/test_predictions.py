from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_dataset_loads():
    data = pd.read_csv(ROOT / "data" / "india_housing_prices.csv")
    assert len(data) > 0
    required = {"State", "City", "Locality", "Price_in_Lakhs"}
    assert required.issubset(set(data.columns))


def test_classifier_pipeline_loads():
    model = joblib.load(ROOT / "models" / "gradient_boosting_model.pkl")
    assert "preprocess" in model.named_steps
    assert "clf" in model.named_steps
    assert model.n_features_in_ == 55


def test_regressor_pipeline_loads():
    model = joblib.load(ROOT / "models" / "gradient_boosting_regressor.pkl")
    assert "preprocess" in model.named_steps
    assert "regressor" in model.named_steps
    assert model.n_features_in_ == 55


@pytest.fixture
def inference_context():
    source = pd.read_csv(ROOT / "data" / "india_housing_prices.csv")
    classifier = joblib.load(ROOT / "models" / "gradient_boosting_model.pkl")
    regressor = joblib.load(ROOT / "models" / "gradient_boosting_regressor.pkl")
    # Load only the inference helpers so tests do not execute Streamlit's UI.
    import ast

    app_source = (ROOT / "streamlit_app" / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(app_source)
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name in {"derived_features", "prediction_frame"}]
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {"pd": pd, "np": np}
    exec(compile(module, "streamlit_app/app.py", "exec"), namespace)
    namespace["prediction_frame"]._preprocessor = classifier.named_steps["preprocess"]
    fields = ["State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
              "Year_Built", "Furnished_Status", "Floor_No", "Total_Floors", "Nearby_Schools",
              "Nearby_Hospitals", "Public_Transport_Accessibility", "Parking_Space", "Security",
              "Amenities", "Facing", "Owner_Type", "Availability_Status"]
    return source, classifier, regressor, namespace["prediction_frame"], fields


def test_saved_models_predict_for_form_inputs(inference_context):
    source, classifier, regressor, make_frame, fields = inference_context
    for index, overrides in [
        (0, {"BHK": 2, "Size_in_SqFt": 850.0, "Price_in_Lakhs": 75.0}),
        (source["Property_Type"].eq("Villa").idxmax(),
         {"BHK": 4, "Size_in_SqFt": 2400.0, "Price_in_Lakhs": 260.0}),
        (len(source) - 1, {"BHK": 3, "Size_in_SqFt": 1500.0, "Price_in_Lakhs": 120.0}),
    ]:
        record = source.iloc[index]
        data = {column: record[column] for column in fields}
        data.update(overrides)
        data["Year_Built"] = 2021
        features = make_frame(data, source, classifier.feature_names_in_)
        prediction = classifier.predict(features)
        probabilities = classifier.predict_proba(features)
        future_value = regressor.predict(features)
        assert prediction.shape == (1,)
        assert probabilities.shape == (1, 2)
        assert np.isfinite(probabilities).all()
        assert np.isfinite(future_value).all()
        assert np.isfinite((future_value[0] - data["Price_in_Lakhs"]) / data["Price_in_Lakhs"])


def test_unseen_and_missing_categories_are_safe(inference_context):
    source, classifier, regressor, make_frame, fields = inference_context
    record = source.iloc[0]
    data = {column: record[column] for column in fields}
    data.update({"City": "New Unseen City", "Locality": "Unseen Locality", "Amenities": None,
                 "BHK": None, "Price_in_Lakhs": 100.0, "Size_in_SqFt": 1200.0,
                 "Year_Built": 2020, "Furnished_Status": None,
                 "Public_Transport_Accessibility": None})
    features = make_frame(data, source, classifier.feature_names_in_)
    assert np.isfinite(classifier.predict(features)).all()
    assert np.isfinite(classifier.predict_proba(features)).all()
    assert np.isfinite(regressor.predict(features)).all()


def test_original_mixed_category_input_is_normalized(inference_context):
    source, classifier, _, make_frame, fields = inference_context
    record = source.iloc[0]
    data = {column: record[column] for column in fields}
    data["Amenities"] = 3.0  # Original failure: numeric count compared with fitted string categories.
    data["BHK"], data["Size_in_SqFt"], data["Price_in_Lakhs"] = 3, 1500.0, 100.0
    data["Year_Built"] = 2021
    features = make_frame(data, source, classifier.feature_names_in_)
    assert classifier.predict(features).shape == (1,)
