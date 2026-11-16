
import os
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Mobile Price Prediction AI",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==================================================
# 2. CUSTOM CSS
# ==================================================

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

.hero-card {
    background: linear-gradient(120deg, rgba(30,41,90,0.85), rgba(20,25,55,0.75));
    border: 1px solid rgba(129,140,248,0.3);
    border-radius: 25px;
    padding: 45px 38px;
    margin-bottom: 30px;
    box-shadow: 0 15px 50px rgba(0,0,0,0.25);
}

.hero-tag {
    display: inline-block;
    padding: 8px 15px;
    border-radius: 30px;
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.35);
    color: #67e8f9;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 20px;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(38px, 6vw, 65px);
    font-weight: 700;
    line-height: 1.08;
    color: white;
    margin-bottom: 18px;
}

.hero-title span {
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-desc {
    color: #cbd5e1;
    font-size: 16px;
    line-height: 1.8;
    max-width: 700px;
}

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 24px;
    font-weight: 700;
    color: #f8fafc;
    margin: 25px 0 8px 0;
}

.section-desc {
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 20px;
}

.input-heading {
    color: #67e8f9;
    font-size: 15px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 8px;
}

div[data-testid="stNumberInput"] label {
    color: #e2e8f0 !important;
    font-weight: 500;
}

div[data-testid="stNumberInput"] input {
    background: #111a32 !important;
    color: white !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
}

div[data-testid="stNumberInput"] button {
    background: #1e293b !important;
    color: #67e8f9 !important;
    border: none !important;
}

div[data-testid="stForm"] {
    background: rgba(15,23,42,0.65);
    border: 1px solid rgba(100,116,139,0.3);
    border-radius: 20px;
    padding: 25px;
}

.stFormSubmitButton button {
    background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
    letter-spacing: 1px;
    padding: 15px !important;
    min-height: 52px;
    transition: 0.3s;
}

.stFormSubmitButton button:hover {
    box-shadow: 0 0 25px rgba(99,102,241,0.5);
    transform: translateY(-2px);
}

.result-card {
    background: linear-gradient(120deg, rgba(30,64,175,0.35), rgba(88,28,135,0.35));
    border: 1px solid rgba(129,140,248,0.5);
    border-radius: 22px;
    padding: 30px;
    text-align: center;
    margin-top: 30px;
    box-shadow: 0 10px 40px rgba(37,99,235,0.15);
}

.result-label {
    color: #a5b4fc;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 15px;
}

.result-value {
    color: #67e8f9;
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(35px, 6vw, 55px);
    font-weight: 800;
    margin: 10px 0;
    overflow-wrap: anywhere;
}

.result-note {
    color: #cbd5e1;
    font-size: 14px;
    margin-top: 12px;
}

.spec-card {
    background: rgba(15,23,42,0.8);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 15px;
    margin: 5px 0;
}

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 13px;
    margin-top: 55px;
    padding-top: 20px;
    border-top: 1px solid rgba(148,163,184,0.2);
}

.footer strong {
    color: #67e8f9;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. HERO SECTION
# ==================================================

st.markdown("""
<div class="hero-card">

    <div class="hero-tag">
        MACHINE LEARNING PROJECT
    </div>

    <div class="hero-title">
        Mobile Price<br>
        <span>Prediction AI</span>
    </div>

    <div class="hero-desc">
        Discover the predicted price category of a mobile phone
        using its hardware specifications and a Support Vector
        Machine classification model.
    </div>

</div>
""", unsafe_allow_html=True)


# ==================================================
# 4. DATASET CONFIGURATION
# ==================================================

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

DATASET_OPTIONS = [
    "mobile_price_dataset.csv",
    "mobile_price_training_300.csv"
]


# ==================================================
# 5. LOAD DATASET AND TRAIN MODEL
# ==================================================

@st.cache_resource
def train_model():

    # Automatically find the available dataset file
    dataset_file = None

    for filename in DATASET_OPTIONS:
        if os.path.exists(filename):
            dataset_file = filename
            break

    if dataset_file is None:
        raise FileNotFoundError(
            "Dataset not found. Upload either "
            "mobile_price_dataset.csv or "
            "mobile_price_training_300.csv."
        )

    df = pd.read_csv(dataset_file)

    # Clean column names
    df.columns = df.columns.str.strip()

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing dataset columns: "
            + ", ".join(missing_columns)
        )

    df = df[required_columns].copy()

    # Convert ONLY input features to numeric
    for col in FEATURES:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Keep target values as text categories
    df[TARGET] = df[TARGET].astype("string").str.strip()

    # Remove invalid rows
    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna(
        subset=FEATURES + [TARGET]
    )

    # Remove empty target labels
    df = df[df[TARGET] != ""]

    if df.empty:
        raise ValueError(
            "No valid rows remain. Check the dataset values."
        )

    if df[TARGET].nunique() < 2:
        raise ValueError(
            "The target must contain at least two price categories."
        )

    # Separate input and target
    X = df[FEATURES]
    y = df[TARGET].astype(str)

    # Main machine learning model: SVM
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced"
        ))
    ])

    # Train model on the complete cleaned dataset
    model.fit(X, y)

    return model, df, dataset_file


# ==================================================
# 6. INITIALIZE MODEL
# ==================================================

try:

    with st.spinner("Training AI model..."):
        model, df, dataset_file = train_model()

    st.success(
        f"AI model successfully trained using {dataset_file}"
    )

except Exception as e:

    st.error(f"Model initialization failed: {e}")

    st.info(
        "Check that your CSV file is uploaded in the "
        "same GitHub repository as app.py."
    )

    st.stop()


# ==================================================
# 7. INPUT LIMIT FUNCTION
# ==================================================

def get_limits(column):

    minimum = float(df[column].min())
    maximum = float(df[column].max())
    median = float(df[column].median())

    if minimum == maximum:
        maximum = minimum + 1

    return minimum, maximum, median


# ==================================================
# 8. INPUT SECTION
# ==================================================

st.markdown("""
<div class="section-title">
    Enter Mobile Specifications
</div>

<div class="section-desc">
    Fill in the hardware specifications below and let the AI model
    predict the mobile phone's price category.
</div>
""", unsafe_allow_html=True)


# ==================================================
# 9. PREDICTION FORM
# ==================================================

with st.form("mobile_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<div class="input-heading">📷 CAMERA SPECIFICATIONS</div>',
            unsafe_allow_html=True
        )

        low, high, default = get_limits(
            "Rear_Main_Camera_MP"
        )

        rear_camera = st.number_input(
            "Rear Main Camera (MP)",
            min_value=low,
            max_value=high,
            value=default,
            step=1.0
        )

        low, high, default = get_limits(
            "Rear_Camera_Count"
        )

        rear_count = st.number_input(
            "Number of Rear Cameras",
            min_value=int(np.floor(low)),
            max_value=int(np.ceil(high)),
            value=int(round(default)),
            step=1
        )

        low, high, default = get_limits(
            "Front_Camera_MP"
        )

        front_camera = st.number_input(
            "Front Camera (MP)",
            min_value=low,
            max_value=high,
            value=default,
            step=1.0
        )

        st.markdown(
            '<div class="input-heading">💾 MEMORY & STORAGE</div>',
            unsafe_allow_html=True
        )

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

        st.markdown(
            '<div class="input-heading">🔋 BATTERY SPECIFICATIONS</div>',
            unsafe_allow_html=True
        )

        low, high, default = get_limits("Battery_mAh")

        battery = st.number_input(
            "Battery Capacity (mAh)",
            min_value=low,
            max_value=high,
            value=default,
            step=100.0
        )

        st.markdown(
            '<div class="input-heading">📱 DISPLAY SPECIFICATIONS</div>',
            unsafe_allow_html=True
        )

        low, high, default = get_limits(
            "Display_Size_inches"
        )

        display = st.number_input(
            "Display Size (inches)",
            min_value=low,
            max_value=high,
            value=default,
            step=0.1
        )

    st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "⚡ GENERATE AI PRICE PREDICTION",
        use_container_width=True
    )


# ==================================================
# 10. GENERATE PREDICTION
# ==================================================

if submitted:

    user_data = pd.DataFrame([{
        "Rear_Main_Camera_MP": rear_camera,
        "Rear_Camera_Count": rear_count,
        "Front_Camera_MP": front_camera,
        "RAM_GB": ram,
        "Storage_GB": storage,
        "Battery_mAh": battery,
        "Display_Size_inches": display
    }], columns=FEATURES)

    try:

        prediction = model.predict(user_data)[0]

        # Save prediction in session state
        st.session_state["prediction"] = str(prediction)

        st.session_state["user_specs"] = user_data.copy()

    except Exception as e:

        st.error(f"Prediction failed: {e}")


# ==================================================
# 11. DISPLAY RESULT
# ==================================================

if "prediction" in st.session_state:

    prediction = st.session_state["prediction"]

    st.markdown(f"""
    <div class="result-card">

        <div class="result-label">
            PREDICTED PRICE RANGE (INR)
        </div>

        <div class="result-value">
            ₹ {prediction}
        </div>

        <div class="result-note">
            Prediction generated using Support Vector Machine
            Classification.
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-title">
        Your Entered Specifications
    </div>
    """, unsafe_allow_html=True)

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

    spec_columns = st.columns(2)

    for index, feature in enumerate(FEATURES):

        value = saved_specs.iloc[0][feature]
        unit = units[feature]

        with spec_columns[index % 2]:

            st.markdown(
                f"""
                <div class="spec-card">
                    <div style="color:#94a3b8;font-size:12px;">
                        {spec_names[feature]}
                    </div>

                    <div style="color:#f8fafc;font-size:20px;font-weight:700;">
                        {value:g} {unit}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ==================================================
# 12. FOOTER
# ==================================================

st.markdown("""
<div class="footer">

    <div>
        <strong>Mobile Price Prediction AI</strong>
    </div>

    <div style="margin-top:8px;">
        Machine Learning Project | Support Vector Machine
    </div>

    <div style="margin-top:12px;">
        App Developed By
        <strong>Tushar, Kishan, Aman, Rajdeep & Ayush</strong>
    </div>

</div>
""", unsafe_allow_html=True)
