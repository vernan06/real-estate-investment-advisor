"""Real Estate Investment Advisor Streamlit application."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "india_housing_prices.csv"
CLASSIFIER_PATH = ROOT / "models" / "gradient_boosting_model.pkl"
REGRESSOR_PATH = ROOT / "models" / "gradient_boosting_regressor.pkl"
APP_TITLE = "Real Estate Investment Advisor: Predicting Property Profitability & Future Value"

st.set_page_config(page_title=APP_TITLE, page_icon="HOUSE", layout="wide", initial_sidebar_state="collapsed")


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models() -> tuple:
    classifier = joblib.load(CLASSIFIER_PATH)
    regressor = joblib.load(REGRESSOR_PATH) if REGRESSOR_PATH.exists() else None
    return classifier, regressor


def categories(df: pd.DataFrame, column: str) -> list:
    return sorted(df[column].dropna().unique().tolist())


def derived_features(df: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    numeric_columns = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Year_Built", "Floor_No",
                       "Total_Floors", "Nearby_Schools", "Nearby_Hospitals"]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in numeric_columns:
        if result[column].isna().any():
            result[column] = result[column].fillna(source[column].median())
    result["Price_per_SqFt"] = result["Price_in_Lakhs"] / result["Size_in_SqFt"]
    result["Age_of_Property"] = 2026 - result["Year_Built"]
    result["School_Category"] = pd.cut(
        result["Nearby_Schools"],
        bins=[-1, 0, 2, 4, 6, np.inf],
        labels=["No Schools", "1-2 Schools", "3-4 Schools", "5-6 Schools", "7+ Schools"],
    )
    result["Hospital_Category"] = pd.cut(
        result["Nearby_Hospitals"],
        bins=[-1, 0, 2, 4, 6, np.inf],
        labels=["No Hospitals", "1-2 Hospitals", "3-4 Hospitals", "5-6 Hospitals", "7+ Hospitals"],
    )
    result["Combined_Amenities"] = result["Nearby_Schools"] + result["Nearby_Hospitals"]
    combined = result["Combined_Amenities"]
    result["Combined_Category"] = np.select(
        [combined <= 2, combined <= 5, combined <= 8],
        ["Low (0-2)", "Medium (3-5)", "High (6-8)"],
        default="Very High (9+)",
    )
    result["Parking_Space_Numeric"] = result["Parking_Space"].eq("Yes").astype(int)
    result["Price_per_SqFt_log"] = np.log1p(result["Price_per_SqFt"])
    source_ppsf = source["Price_per_SqFt"].dropna()
    result["Price_per_SqFt_winsor"] = result["Price_per_SqFt"].clip(
        source_ppsf.quantile(0.01), source_ppsf.quantile(0.99)
    )
    result["City_freq"] = source["City"].value_counts(normalize=True).to_dict().get(result["City"].iloc[0], 1e-6)
    result["Locality_freq"] = source["Locality"].value_counts(normalize=True).to_dict().get(result["Locality"].iloc[0], 1e-6)
    result["Price_Range"] = pd.cut(
        result["Price_in_Lakhs"],
        bins=[-np.inf, 50, 100, 200, np.inf],
        labels=["Low", "Medium", "High", "Premium"],
    )
    furnished_map = {"Unfurnished": 0, "Semi-furnished": 1, "Furnished": 2}
    transport_map = {"Low": 0, "Medium": 1, "High": 2}
    result["Furnished_Status"] = result["Furnished_Status"].astype("string").map(furnished_map).fillna(0)
    result["Public_Transport_Accessibility"] = result["Public_Transport_Accessibility"].astype("string").map(transport_map).fillna(0)
    result["Investment_Score"] = (
        result["BHK"].ge(3).astype(int)
        + result["Furnished_Status"].ge(1).astype(int)
        + result["Availability_Status"].ne("Under_Construction").astype(int)
    )

    model_row = {}
    expected = list(getattr(getattr(source, "_classifier", None), "feature_names_in_", []))
    for column in [
        "City", "Locality", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
        "Year_Built", "Furnished_Status", "Floor_No", "Total_Floors", "Age_of_Property",
        "Nearby_Schools", "Nearby_Hospitals", "Public_Transport_Accessibility", "Amenities",
        "Price_Range", "School_Category", "Hospital_Category", "Combined_Amenities",
        "Combined_Category", "Parking_Space_Numeric", "Price_per_SqFt_log",
        "Price_per_SqFt_winsor", "City_freq", "Locality_freq", "Investment_Score",
    ]:
        model_row[column] = result[column].iloc[0]
    model_row["Price_Range"] = result["Price_Range"].iloc[0]
    model_row["School_Category"] = result["School_Category"].iloc[0]
    model_row["Hospital_Category"] = result["Hospital_Category"].iloc[0]
    model_row["Combined_Category"] = result["Combined_Category"].iloc[0]
    model_row["Amenities"] = result["Amenities"].iloc[0]

    for column in ["State", "Property_Type", "Parking_Space", "Security", "Facing", "Owner_Type", "Availability_Status"]:
        prefix = column + "_"
        for value in sorted(source[column].dropna().unique().tolist()[1:]):
            model_row[prefix + str(value)] = int(result[column].iloc[0] == value)
    row = pd.DataFrame([model_row])
    expected_columns = list(getattr(derived_features, "_expected_columns", row.columns))
    expected_columns = list(dict.fromkeys(expected_columns))
    fitted_preprocessor = getattr(derived_features, "_preprocessor", None)
    if fitted_preprocessor is not None:
        onehot = fitted_preprocessor.named_transformers_["onehot"]
        categorical_columns = ["City", "Locality", "Amenities", "School_Category", "Hospital_Category", "Combined_Category"]
        categories = dict(zip(categorical_columns, onehot.categories_))
        for column in categorical_columns:
            value = row[column].iloc[0]
            if pd.isna(value):
                row[column] = "__unknown__"
            else:
                row[column] = str(value)
            if row[column].iloc[0] not in categories[column]:
                row[column] = "__unknown__"
    numeric_columns = row.columns.difference(["City", "Locality", "Amenities", "School_Category", "Hospital_Category", "Combined_Category"])
    for column in numeric_columns:
        row[column] = pd.to_numeric(row[column], errors="coerce")
    return row.reindex(columns=expected_columns, fill_value=0)


def prediction_frame(data: dict, source: pd.DataFrame, expected_columns) -> pd.DataFrame:
    row = pd.DataFrame([data])
    derived_features._expected_columns = list(expected_columns)
    derived_features._preprocessor = getattr(prediction_frame, "_preprocessor", None)
    return derived_features(row, source)


st.markdown("""
<style>
:root {
  --app-bg:#F7F8FA; --card-bg:#FFFFFF; --text:#172033; --muted:#526176;
  --primary:#176B62; --secondary:#D97757; --border:#DDE3EA;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"] {
  background:var(--app-bg) !important; color:var(--text) !important;
}
.block-container { max-width:1360px; padding:1.5rem 2rem 3rem !important; }
header[data-testid="stHeader"] { background:var(--app-bg) !important; }
.hero { padding:.25rem 0 1rem; border-bottom:1px solid var(--border); margin-bottom:.75rem; }
.hero h1 { color:var(--text); font-size:clamp(1.65rem,3vw,2.35rem); margin:0; letter-spacing:-.03em; }
.hero p { color:var(--muted); font-size:1rem; margin:.35rem 0 0; }
.card { min-height:112px; background:var(--card-bg); border:1px solid var(--border); border-radius:12px;
  padding:1.1rem 1.25rem; box-shadow:0 2px 8px rgba(23,32,51,.05); }
.metric-label { color:var(--muted); font-size:.82rem; font-weight:700; letter-spacing:.02em; text-transform:uppercase; }
.metric-value { color:var(--text); font-size:clamp(1.45rem,2.2vw,2rem); font-weight:750; line-height:1.15;
  margin-top:.55rem; font-variant-numeric:tabular-nums; }
.chart-card { background:var(--card-bg); border:1px solid var(--border); border-radius:12px;
  padding:.8rem .8rem .25rem; margin-bottom:1rem; box-shadow:0 2px 8px rgba(23,32,51,.04); }
.chart-card h4 { color:var(--text) !important; margin:.25rem .35rem .25rem !important; }
.good { color:#137A46; font-size:1.5rem; font-weight:750; }
.risk { color:#B42318; font-size:1.5rem; font-weight:750; }
div[data-testid="stMarkdownContainer"] p, div[data-testid="stMarkdownContainer"] li { color:var(--text); }
label, .stMarkdown, .stCaption, small { color:var(--text) !important; }
div[data-baseweb="tab-list"] { border-bottom:1px solid var(--border); gap:.25rem; overflow-x:auto !important; flex-wrap:nowrap !important; }
div[data-baseweb="tab-list"] button { flex:0 0 auto !important; white-space:nowrap !important; }
button[data-baseweb="tab"] { color:var(--muted) !important; font-weight:650 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color:var(--primary) !important; border-bottom:3px solid var(--primary) !important; }
div[data-baseweb="select"] { background:var(--card-bg) !important; }
.stButton button { background:var(--primary) !important; color:#FFFFFF !important; border:1px solid var(--primary) !important;
  border-radius:8px; min-height:2.75rem; font-weight:700; }
.stButton button:hover { background:#115B54 !important; color:#FFFFFF !important; }
.stAlert { border-radius:10px; }
.stDeployButton { display:none; }
@media (max-width:700px) {
  .block-container { padding:1rem .75rem 2rem !important; }
  .hero h1 { font-size:1.55rem; }
  .card { min-height:96px; }
}
</style>
""", unsafe_allow_html=True)


def style_chart(figure: go.Figure) -> go.Figure:
    """Apply the light application theme without changing chart data."""
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="DM Sans, Arial, sans-serif", color="#172033", size=12),
        margin=dict(l=55, r=20, t=48, b=55),
        title_font=dict(color="#172033", size=16),
        hoverlabel=dict(bgcolor="#FFFFFF", font_color="#172033", bordercolor="#DDE3EA"),
        legend=dict(font=dict(color="#172033")),
        colorway=["#176B62", "#D97757", "#4776A8", "#8A6FA3", "#4F7D67", "#C6922E"],
    )
    figure.update_xaxes(color="#526176", tickfont=dict(color="#526176"), title_font=dict(color="#172033"),
                        gridcolor="#E6EAF0", linecolor="#C8D0DB")
    figure.update_yaxes(color="#526176", tickfont=dict(color="#526176"), title_font=dict(color="#172033"),
                        gridcolor="#E6EAF0", linecolor="#C8D0DB")
    for trace in figure.data:
        if hasattr(trace, "marker"):
            if isinstance(trace.marker.color, (list, tuple, np.ndarray, pd.Series)):
                continue
            trace.marker.color = "#176B62"
    return figure

df = load_data()
classifier, regressor = load_models()
expected_columns = list(classifier.feature_names_in_)
prediction_frame._preprocessor = classifier.named_steps["preprocess"]

st.markdown(f'<div class="hero"><h1>{APP_TITLE}</h1></div>', unsafe_allow_html=True)

tabs = st.tabs(["Dashboard", "Investment Advisor & Forecast", "Property Analytics", "Model Performance", "About / Methodology"])

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    for column, label, value in zip(
        [c1, c2, c3, c4],
        ["Properties", "States", "Cities", "Median price"],
        [f"{len(df):,}", f"{df['State'].nunique()}", f"{df['City'].nunique()}", f"{df['Price_in_Lakhs'].median():,.2f} lakh"],
    ):
        column.markdown(
            f'<div class="card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(style_chart(px.histogram(df, x="Price_in_Lakhs", nbins=60,
                                               title="Distribution of property prices")), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        city = df.groupby("City", as_index=False)["Price_in_Lakhs"].mean().sort_values("Price_in_Lakhs", ascending=False).head(10)
        st.plotly_chart(style_chart(px.bar(city, x="City", y="Price_in_Lakhs",
                                           title="Average price by city")), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    col1, col2 = st.columns(2)
    with col1:
        state = st.selectbox("State", categories(df, "State"))
        state_cities = sorted(df.loc[df["State"] == state, "City"].dropna().unique())
        city = st.selectbox("City", state_cities)
        localities = sorted(df.loc[(df["State"] == state) & (df["City"] == city), "Locality"].unique())
        locality = st.selectbox("Locality", localities)
        property_type = st.selectbox("Property type", categories(df, "Property_Type"))
        bhk = st.number_input("BHK", 1, 10, 3)
        area = st.number_input("Area (sq ft)", 100.0, 20_000.0, 1_500.0, step=100.0)
        price = st.number_input("Current price (lakhs)", 1.0, 10_000.0, 100.0, step=5.0)
        age = st.slider("Property age (years)", 0, 100, 5)
    with col2:
        furnishing = st.selectbox("Furnishing status", categories(df, "Furnished_Status"))
        parking = st.selectbox("Parking", categories(df, "Parking_Space"))
        amenities = st.text_input("Amenities", "Gym, Garden, Security")
        schools = st.number_input("Nearby schools", 0, 20, 4)
        hospitals = st.number_input("Nearby hospitals", 0, 20, 3)
        transport = st.selectbox("Public transport accessibility", categories(df, "Public_Transport_Accessibility"))
        floor = st.number_input("Floor", 0, 100, 2)
        total_floors = st.number_input("Total floors", 1, 100, 10)
        security = st.selectbox("Security", categories(df, "Security"))
        facing = st.selectbox("Facing", categories(df, "Facing"))
        owner = st.selectbox("Owner type", categories(df, "Owner_Type"))
        availability = st.selectbox("Availability", categories(df, "Availability_Status"))

    if st.button("Analyze investment", type="primary"):
        data = {
            "State": state, "City": city, "Locality": locality, "Property_Type": property_type,
            "BHK": bhk, "Size_in_SqFt": area, "Price_in_Lakhs": price, "Year_Built": 2026 - age,
            "Furnished_Status": furnishing, "Floor_No": floor, "Total_Floors": total_floors,
            "Nearby_Schools": schools, "Nearby_Hospitals": hospitals,
            "Public_Transport_Accessibility": transport, "Parking_Space": parking,
            "Security": security, "Amenities": amenities, "Facing": facing, "Owner_Type": owner,
            "Availability_Status": availability,
        }
        try:
            features = prediction_frame(data, df, expected_columns)
            prediction = int(classifier.predict(features)[0])
            probability = classifier.predict_proba(features)[0]
            future = None if regressor is None else float(regressor.predict(features)[0])
            a, b = st.columns(2)
            with a:
                label = "Good Investment" if prediction else "Not a Good Investment"
                st.markdown(f'<div class="card"><div class="{"good" if prediction else "risk"}">{label}</div><p>Model confidence: {max(probability):.1%}</p></div>', unsafe_allow_html=True)
            with b:
                if future is not None:
                    st.markdown(f'<div class="card"><b>Estimated value in five years</b><div style="font-size:1.7rem;font-weight:700">{future:,.2f} lakh</div><p>Appreciation: {(future-price)/price:.1%}</p></div>', unsafe_allow_html=True)
                else:
                    st.info("Five-year regression artifact is unavailable.")
        except Exception as exc:
            st.error(f"Prediction could not be generated: {exc}")

with tabs[2]:
    def chart(title, figure):
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown(f"#### {title}")
        st.plotly_chart(style_chart(figure), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    analyses = [
        ("Property price distribution", px.histogram(df, x="Price_in_Lakhs", nbins=60)),
        ("Property size distribution", px.histogram(df, x="Size_in_SqFt", nbins=60)),
        ("Price per sq ft by property type", px.box(df, x="Property_Type", y="Price_per_SqFt")),
        ("Property size versus price", px.scatter(df.sample(min(5000, len(df)), random_state=42), x="Size_in_SqFt", y="Price_in_Lakhs", opacity=.4)),
        ("Price per sq ft outliers", px.box(df, y="Price_per_SqFt")),
        ("Average price per sq ft by state", px.bar(df.groupby("State", as_index=False)["Price_per_SqFt"].mean(), x="State", y="Price_per_SqFt")),
        ("Average property price by city", px.bar(df.groupby("City", as_index=False)["Price_in_Lakhs"].mean().sort_values("Price_in_Lakhs", ascending=False).head(15), x="City", y="Price_in_Lakhs")),
        ("Median property age by locality", px.bar(df.groupby("Locality", as_index=False)["Age_of_Property"].median().sort_values("Age_of_Property", ascending=False).head(15), x="Locality", y="Age_of_Property")),
        ("BHK distribution across cities", px.histogram(df, x="BHK", color="City")),
        ("Most expensive localities (cross-sectional comparison)", px.box(df[df["Locality"].isin(df.groupby("Locality")["Price_in_Lakhs"].median().nlargest(5).index)], x="Locality", y="Price_in_Lakhs")),
        ("Numerical feature correlations", px.imshow(df.select_dtypes(include="number").corr(), aspect="auto")),
        ("Nearby schools versus price per sq ft", px.box(df, x="Nearby_Schools", y="Price_per_SqFt")),
        ("Nearby hospitals versus price per sq ft", px.box(df, x="Nearby_Hospitals", y="Price_per_SqFt")),
        ("Property price by furnishing status", px.box(df, x="Furnished_Status", y="Price_in_Lakhs")),
        ("Price per sq ft by facing direction", px.box(df, x="Facing", y="Price_per_SqFt")),
        ("Properties by owner type", px.histogram(df, x="Owner_Type")),
        ("Properties by availability status", px.histogram(df, x="Availability_Status")),
        ("Parking space versus property price", px.box(df, x="Parking_Space", y="Price_in_Lakhs")),
        ("Amenities versus price per sq ft", px.scatter(df.sample(min(5000, len(df)), random_state=42), x="Amenities", y="Price_per_SqFt", opacity=.4)),
        ("Transport accessibility versus price per sq ft", px.box(df, x="Public_Transport_Accessibility", y="Price_per_SqFt")),
    ]
    for index in range(0, len(analyses), 2):
        c1, c2 = st.columns(2)
        for column, (title, figure) in zip([c1, c2], analyses[index:index + 2]):
            with column:
                chart(title, figure)
    st.caption("The source dataset is cross-sectional and has no observation date. Locality rankings are current dataset comparisons, not historical price trends.")

with tabs[3]:
    st.subheader("Model artifact and evaluation notes")
    st.write("Classification and regression models are serialized sklearn pipelines. The Gradient Boosting artifacts use 55 matching model features.")
    report_path = ROOT / "models" / "rf_report.json"
    if report_path.exists():
        st.json(__import__("json").loads(report_path.read_text()))
    st.info("Do not treat metrics derived from a rule-generated target or assumption-based forecast as real-world accuracy. See train_models.py and the generated training report for training details.")

with tabs[4]:
    st.subheader(APP_TITLE)
    st.markdown("""
This application classifies properties as good or not-good investments and estimates five-year value.
Inputs are preprocessed consistently with the saved pipelines. The classification target and forecast target are
also described in train_models.py; ensure their assumptions are understood before operational use.

**Dataset limitation:** the supplied CSV contains 250,000 cross-sectional records and no historical transaction dates.
Five-year forecasts are therefore based on train_models.py target-generation assumptions, not observed resale history.
""")
