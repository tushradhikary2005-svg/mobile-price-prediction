
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Mobile Price Prediction AI",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =====================================================
# 2. PREMIUM DARK THEME
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

.stApp {
    background:
        radial-gradient(circle at 10% 10%, #172554 0%, transparent 35%),
        radial-gradient(circle at 90% 20%, #312e81 0%, transparent 30%),
        linear-gradient(135deg, #070b18, #10152b 55%, #090d1c);
    color: #f8fafc;
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #f8fafc !important;
}

p, label, span {
    color: #e2e8f0;
}

div[data-testid="stNumberInput"] input {
    background: #111a32 !important;
    color: white !important;
    border: 1px solid #475569 !important;
    border-radius: 10px !important;
}

div[data-testid="stNumberInput"] button {
    background: #1e293b !important;
    color: #67e8f9 !important;
}

div[data-testid="stForm"] {
    background: rgba(15,23,42,0.75);
    border: 1px solid rgba(129,140,248,0.3);
    border-radius: 20px;
    padding: 25px;
}

.stFormSubmitButton button {
    background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
    min-height: 52px;
}

.stFormSubmitButton button:hover {
    box-shadow: 0 0 25px rgba(99,102,241,0.5);
}

div[data-testid="stMetric"] {
    background: rgba(30,41,90,0.65);
    border: 1px solid rgba(129,140,248,0.4);
    border-radius: 16px;
    padding: 22px;
}

div[data-testid="stMetricValue"] {
    color: #67e8f9 !important;
    font-size: 32px !important;
    font-weight: 800 !important;
}

div[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
}

hr {
    border-color: rgba(148,163,184,0.25);
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. HERO SECTION
# =====================================================

st.markdown("### MACHINE LEARNING PROJECT")

st.title("📱 Mobile Price Prediction AI")

st.markdown(
    "Discover the predicted price category of a mobile phone "
    "using its hardware specifications and a Support Vector "
    "Machine classification model."
)

st.divider()


# =====================================================
# 4. DATASET CONFIGURATION
# =====================================================

DATASET_FILE = "mobile_price_dataset.csv"

FEATURES = [
    "Rear_Main_Camera_MP",
    "Rear_Camera_Count",
    "Front_Camera_MP",
    "RAM_GB",
    "Storage_GB",
    "Battery_mAh",
    "Display_Size_inches"
]

TARGET = "Price_Range_INR"


# =====================================================
# 5. LOAD AND CLEAN DATASET
# =====================================================

@st.cache_data
def load_dataset():

    df = pd.read_csv(DATASET_FILE)

    # Remove unwanted spaces from column names
    df.columns = df.columns.astype(str).str.strip()

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing dataset columns: "
            + ", ".join(missing_columns)
            + ". Please check your CSV column names."
        )

    df = df[required_columns].copy()

    # Convert ONLY input features to numeric.
    # Keep target labels as they are.
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean target labels without converting them to numeric
    df[TARGET] = df[TARGET].astype("string").str.strip()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Remove rows with missing input features or target
    df.dropna(subset=FEATURES + [TARGET], inplace=True)

    # Remove empty target labels
    df = df[df[TARGET] != ""]

    if len(df) == 0:
        raise ValueError(
            "No valid rows remain. Check the CSV feature columns "
            "and Price_Range_INR values."
        )

    if df[TARGET].nunique() < 2:
        raise ValueError(
            "The target column must contain at least two different classes."
        )

    return df.reset_index(drop=True)


# =====================================================
# 6. TRAIN SVM MODEL
# =====================================================

@st.cache_resource
def train_model():

    df = load_dataset()

    X = df[FEATURES]
    y = df[TARGET].astype(str)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced"
        ))
    ])

    model.fit(X, y)

    return model, df


# =====================================================
# 7. INITIALIZE MODEL
# =====================================================

try:

    with st.spinner("Training your AI model..."):
        model, df = train_model()

    st.success("AI Model Successfully Initialized!")

except Exception as error:

    st.error("Model initialization failed.")

    st.exception(error)

    st.info(
        "Check that mobile_price_dataset.csv is in the same "
        "GitHub repository as app.py and that its column names "
        "match the required feature names."
    )

    st.stop()


# =====================================================
# 8. DATASET INFORMATION
# =====================================================

st.markdown("### 🤖 AI Model Information")

info1, info2, info3 = st.columns(3)

info1.metric("Training Records", len(df))
info2.metric("Price Categories", df[TARGET].nunique())
info3.metric("Algorithm", "SVM")

st.divider()


# =====================================================
# 9. INPUT LIMITS
# =====================================================

def get_limits(column):

    minimum = float(df[column].min())
    maximum = float(df[column].max())
    default = float(df[column].median())

    if minimum == maximum:
        maximum = minimum + 1

    return minimum, maximum, default


# =====================================================
# 10. INPUT SECTION
# =====================================================

st.header("📱 Enter Mobile Specifications")

st.caption(
    "Enter the hardware specifications to predict the mobile "
    "phone's price category."
)


# =====================================================
# 11. PREDICTION FORM
# =====================================================

with st.form("mobile_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📷 Camera Specifications")

        low, high, default = get_limits("Rear_Main_Camera_MP")

        rear_camera = st.number_input(
            "Rear Main Camera (MP)",
            min_value=low,
            max_value=high,
            value=default,
            step=1.0
        )

        low, high, default = get_limits("Rear_Camera_Count")

        rear_count = st.number_input(
            "Number of Rear Cameras",
            min_value=int(np.ceil(low)),
            max_value=int(np.floor(high)),
            value=int(np.clip(round(default), np.ceil(low), np.floor(high))),
            step=1
        )

        low, high, default = get_limits("Front_Camera_MP")

        front_camera = st.number_input(
            "Front Camera (MP)",
            min_value=low,
            max_value=high,
            value=default,
            step=1.0
        )

        st.subheader("💾 Memory and Storage")

        low, high, default = get_limits("RAM_GB")

        ram = st.number_input(
            "RAM (GB)",
            min_value=low,
            max_value=high,
            value=default,
            step=1.0
        )

        low, high, default = get_limits("Storage_GB")

        storage = st.number_input(
            "Internal Storage (GB)",
            min_value=low,
            max_value=high,
            value=default,
            step=1.0
        )

    with col2:

        st.subheader("🔋 Battery Specifications")

        low, high, default = get_limits("Battery_mAh")

        battery = st.number_input(
            "Battery Capacity (mAh)",
            min_value=low,
            max_value=high,
            value=default,
            step=100.0
        )

        st.subheader("📺 Display Specifications")

        low, high, default = get_limits("Display_Size_inches")

        display = st.number_input(
            "Display Size (inches)",
            min_value=low,
            max_value=high,
            value=default,
            step=0.1
        )

    st.write("")

    submitted = st.form_submit_button(
        "⚡ GENERATE AI PRICE PREDICTION",
        use_container_width=True
    )


# =====================================================
# 12. GENERATE PREDICTION
# =====================================================

if submitted:

    user_data = pd.DataFrame(
        [[
            rear_camera,
            rear_count,
            front_camera,
            ram,
            storage,
            battery,
            display
        ]],
        columns=FEATURES
    )

    try:

        prediction = model.predict(user_data)[0]

        # Save prediction and input specifications
        st.session_state["prediction"] = str(prediction)
        st.session_state["user_specs"] = user_data.copy()

        st.session_state["has_prediction"] = True

    except Exception as error:

        st.error(f"Prediction failed: {error}")

        st.session_state["has_prediction"] = False


# =====================================================
# 13. DISPLAY RESULT
# =====================================================

if st.session_state.get("has_prediction", False):

    prediction = st.session_state["prediction"]

    st.divider()

    st.header("🎯 Prediction Result")

    st.success("Prediction generated successfully!")

    st.metric(
        label="PREDICTED PRICE RANGE (INR)",
        value=f"₹ {prediction}"
    )

    st.caption(
        "Result generated using Support Vector Machine classification."
    )

    st.subheader("Your Entered Specifications")

    saved_specs = st.session_state["user_specs"]

    spec_names = {
        "Rear_Main_Camera_MP": "Rear Main Camera",
        "Rear_Camera_Count": "Rear Camera Count",
        "Front_Camera_MP": "Front Camera",
        "RAM_GB": "RAM",
        "Storage_GB": "Storage",
        "Battery_mAh": "Battery",
        "Display_Size_inches": "Display Size"
    }

    units = {
        "Rear_Main_Camera_MP": "MP",
        "Rear_Camera_Count": "",
        "Front_Camera_MP": "MP",
        "RAM_GB": "GB",
        "Storage_GB": "GB",
        "Battery_mAh": "mAh",
        "Display_Size_inches": "inches"
    }

    result_columns = st.columns(2)

    for index, feature in enumerate(FEATURES):

        value = saved_specs.iloc[0][feature]

        with result_columns[index % 2]:

            st.metric(
                label=spec_names[feature],
                value=f"{value:g} {units[feature]}".strip()
            )


# =====================================================
# 14. FOOTER
# =====================================================

st.divider()

st.markdown(
    "### Mobile Price Prediction AI"
)

st.caption(
    "Machine Learning Project | Support Vector Machine Classification"
)

st.markdown(
    "**App Developed By:** "
    "Tushar, Kishan, Aman, Rajdeep & Ayush"
)

st.caption("Built with Python, Streamlit and Scikit-learn.")
