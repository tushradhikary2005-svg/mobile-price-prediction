
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Mobile Price Prediction AI",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 2. PREMIUM CSS DESIGN
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #060816;
    --panel: #10162b;
    --blue: #38bdf8;
    --purple: #a78bfa;
    --cyan: #67e8f9;
    --muted: #94a3b8;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(37,99,235,0.20), transparent 35%),
        radial-gradient(ellipse at 90% 15%, rgba(124,58,237,0.19), transparent 35%),
        radial-gradient(ellipse at 50% 100%, rgba(8,145,178,0.10), transparent 40%),
        var(--bg);
    color: #f8fafc;
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

#MainMenu, footer, header {
    visibility: hidden;
}

/* HERO SECTION */

.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(
        125deg,
        rgba(15,35,70,0.98),
        rgba(31,24,72,0.98),
        rgba(9,38,59,0.98)
    );
    border: 1px solid rgba(125,170,255,0.28);
    border-radius: 28px;
    padding: 48px 42px;
    margin-bottom: 30px;
    box-shadow:
        0 20px 70px rgba(0,0,0,0.38),
        inset 0 1px 0 rgba(255,255,255,0.08);
}

.hero:after {
    content: "";
    position: absolute;
    width: 280px;
    height: 280px;
    right: -80px;
    top: -110px;
    border-radius: 50%;
    background: rgba(56,189,248,0.14);
    filter: blur(35px);
}

.hero-tag {
    color: #7dd3fc;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-bottom: 18px;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    color: #ffffff;
    font-size: clamp(36px, 5vw, 58px);
    font-weight: 800;
    line-height: 1.08;
    letter-spacing: -2px;
    margin-bottom: 20px;
    text-shadow: 0 0 35px rgba(56,189,248,0.16);
}

.hero-title span {
    background: linear-gradient(90deg, #67e8f9, #60a5fa, #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-desc {
    color: #cbd5e1;
    font-size: 15px;
    line-height: 1.9;
    max-width: 680px;
}

/* TOP LABEL */

.top-label {
    color: #64748b;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 3px;
    text-align: center;
    margin-bottom: 12px;
}

/* SECTION HEADINGS */

.section-heading {
    font-family: 'Space Grotesk', sans-serif;
    color: #f8fafc;
    font-size: 25px;
    font-weight: 700;
    margin-top: 24px;
    margin-bottom: 5px;
}

.section-description {
    color: #94a3b8;
    font-size: 13px;
    margin-bottom: 24px;
    line-height: 1.7;
}

/* INPUT PANELS */

.input-panel {
    background: linear-gradient(
        145deg,
        rgba(20,31,56,0.94),
        rgba(12,20,39,0.96)
    );
    border: 1px solid rgba(100,140,200,0.22);
    border-radius: 20px;
    padding: 22px;
    margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.16);
}

.panel-heading {
    color: #93c5fd;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 16px;
}

/* INPUT LABELS */

div[data-testid="stNumberInput"] label {
    color: #dbeafe !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

div[data-testid="stNumberInput"] input {
    background: #090f20 !important;
    color: #ffffff !important;
    border: 1px solid #344766 !important;
    border-radius: 11px !important;
    min-height: 43px;
}

div[data-testid="stNumberInput"] input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.15) !important;
}

div[data-testid="stNumberInput"] button {
    color: #7dd3fc !important;
}

/* PREDICTION BUTTON */

.stFormSubmitButton > button {
    background: linear-gradient(100deg, #2563eb, #7c3aed, #0891b2);
    background-size: 200% 200%;
    color: #ffffff !important;
    border: 1px solid rgba(147,197,253,0.35) !important;
    border-radius: 14px !important;
    min-height: 58px;
    font-size: 15px !important;
    font-weight: 800 !important;
    letter-spacing: 1px;
    box-shadow: 0 8px 30px rgba(59,130,246,0.25);
    transition: all 0.25s ease;
}

.stFormSubmitButton > button:hover {
    background-position: 100% 50%;
    transform: translateY(-3px);
    box-shadow: 0 12px 35px rgba(124,58,237,0.42);
    border-color: #67e8f9 !important;
}

/* RESULT CARD */

.result-card {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at 50% 0%, rgba(56,189,248,0.18), transparent 50%),
        linear-gradient(145deg, #102747, #171b40, #101c35);
    border: 1px solid rgba(103,232,249,0.50);
    border-radius: 25px;
    padding: 38px 20px;
    margin-top: 18px;
    text-align: center;
    box-shadow:
        0 15px 50px rgba(37,99,235,0.17),
        inset 0 1px 0 rgba(255,255,255,0.08);
}

.result-tag {
    color: #7dd3fc;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: 4px;
}

.result-title {
    color: #ffffff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 17px;
    font-weight: 600;
    margin-top: 15px;
}

.result-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(38px, 7vw, 62px);
    font-weight: 800;
    margin: 15px 0;
    background: linear-gradient(90deg, #67e8f9, #60a5fa, #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    overflow-wrap: anywhere;
}

.result-note {
    color: #a5b4fc;
    font-size: 13px;
}

/* INFORMATION BOX */

div[data-testid="stExpander"] {
    background: rgba(17,28,50,0.85);
    border: 1px solid rgba(100,140,200,0.22);
    border-radius: 14px;
}

/* FOOTER */

.footer {
    border-top: 1px solid rgba(148,163,184,0.15);
    text-align: center;
    color: #64748b;
    font-size: 11px;
    line-height: 2;
    letter-spacing: 1px;
    padding-top: 28px;
    margin-top: 45px;
}

/* ALERTS */

div[data-testid="stAlert"] {
    border-radius: 13px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. HERO SECTION - YOUR REQUESTED TOP CONTENT
# ============================================================

st.markdown("""
<div class="hero">

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


# ============================================================
# 4. DATASET CONFIGURATION
# ============================================================

DATA_FILE = "mobile_price_dataset.csv"

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


# ============================================================
# 5. SVM MODEL TRAINING
# ============================================================

@st.cache_resource
def train_model():

    df = pd.read_csv(DATA_FILE)

    df = df.loc[
        :,
        ~df.columns.str.contains("^Unnamed")
    ]

    missing = [
        col for col in FEATURES + [TARGET]
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}"
        )

    df = df.dropna(subset=FEATURES + [TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    if y.nunique() < 2:
        raise ValueError(
            "Dataset must contain at least two price classes."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    return model, df


# ============================================================
# 6. LOAD MODEL
# ============================================================

try:

    with st.spinner("INITIALIZING PREDICTION ENGINE..."):
        model, df = train_model()

except FileNotFoundError:

    st.error(
        "Dataset not found. Please upload "
        "mobile_price_dataset.csv to your GitHub repository."
    )

    st.stop()

except Exception as e:

    st.error(f"Model initialization error: {e}")

    st.stop()


# ============================================================
# 7. FEATURE LIMITS
# ============================================================

def get_limits(feature):

    values = pd.to_numeric(
        df[feature],
        errors="coerce"
    ).dropna()

    if values.empty:
        raise ValueError(
            f"No numeric data available for {feature}."
        )

    return (
        float(values.min()),
        float(values.max()),
        float(values.median())
    )


# ============================================================
# 8. INPUT SECTION
# ============================================================

st.markdown("""
<div class="section-heading">
    ⚡ Configure Your Device
</div>

<div class="section-description">
    Enter the hardware specifications of your mobile phone.
    Our AI model will analyze the details and predict its price category.
</div>
""", unsafe_allow_html=True)


# ============================================================
# 9. INPUT FORM
# ============================================================

with st.form("mobile_prediction_form"):

    col1, col2 = st.columns(2, gap="large")

    with col1:

        st.markdown("""
        <div class="input-panel">
            <div class="panel-heading">
                📸 &nbsp; CAMERA SYSTEM
            </div>
        </div>
        """, unsafe_allow_html=True)

        rear_min, rear_max, rear_default = get_limits(
            "Rear_Main_Camera_MP"
        )

        rear_camera = st.number_input(
            "Rear Main Camera (MP)",
            min_value=rear_min,
            max_value=rear_max,
            value=rear_default
        )

        count_min, count_max, count_default = get_limits(
            "Rear_Camera_Count"
        )

        rear_count = st.number_input(
            "Number of Rear Cameras",
            min_value=int(count_min),
            max_value=int(count_max),
            value=int(count_default),
            step=1
        )

        front_min, front_max, front_default = get_limits(
            "Front_Camera_MP"
        )

        front_camera = st.number_input(
            "Front Camera (MP)",
            min_value=front_min,
            max_value=front_max,
            value=front_default
        )

        st.markdown("""
        <div class="input-panel">
            <div class="panel-heading">
                🧠 &nbsp; MEMORY CONFIGURATION
            </div>
        </div>
        """, unsafe_allow_html=True)

        ram_min, ram_max, ram_default = get_limits("RAM_GB")

        ram = st.number_input(
            "RAM (GB)",
            min_value=ram_min,
            max_value=ram_max,
            value=ram_default
        )

    with col2:

        st.markdown("""
        <div class="input-panel">
            <div class="panel-heading">
                💾 &nbsp; INTERNAL STORAGE
            </div>
        </div>
        """, unsafe_allow_html=True)

        storage_min, storage_max, storage_default = get_limits(
            "Storage_GB"
        )

        storage = st.number_input(
            "Storage Capacity (GB)",
            min_value=storage_min,
            max_value=storage_max,
            value=storage_default
        )

        st.markdown("""
        <div class="input-panel">
            <div class="panel-heading">
                🔋 &nbsp; BATTERY SYSTEM
            </div>
        </div>
        """, unsafe_allow_html=True)

        battery_min, battery_max, battery_default = get_limits(
            "Battery_mAh"
        )

        battery = st.number_input(
            "Battery Capacity (mAh)",
            min_value=battery_min,
            max_value=battery_max,
            value=battery_default
        )

        st.markdown("""
        <div class="input-panel">
            <div class="panel-heading">
                📱 &nbsp; DISPLAY TECHNOLOGY
            </div>
        </div>
        """, unsafe_allow_html=True)

        display_min, display_max, display_default = get_limits(
            "Display_Size_inches"
        )

        display = st.number_input(
            "Display Size (inches)",
            min_value=display_min,
            max_value=display_max,
            value=display_default
        )

    st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "✦  GENERATE AI PRICE PREDICTION  ✦",
        use_container_width=True
    )


# ============================================================
# 10. PREDICTION ENGINE
# ============================================================

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

    with st.spinner("ANALYZING DEVICE SPECIFICATIONS..."):

        prediction = model.predict(user_data)[0]

    # --------------------------------------------------------
    # RESULT DISPLAY
    # --------------------------------------------------------

    st.markdown("""
    <div class="section-heading">
        ✨ AI Prediction Result
    </div>

    <div class="section-description">
        Your device has been successfully analyzed.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="result-card">

            <div class="result-tag">
                PREDICTION GENERATED
            </div>

            <div class="result-title">
                Predicted Price Range (INR)
            </div>

            <div class="result-value">
                {prediction}
            </div>

            <div class="result-note">
                SUPPORT VECTOR MACHINE CLASSIFICATION
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.success(
        "Prediction completed successfully!"
    )

    # --------------------------------------------------------
    # USER INPUT SUMMARY
    # --------------------------------------------------------

    with st.expander("📋 VIEW DEVICE SPECIFICATIONS"):

        display_data = user_data.T.reset_index()

        display_data.columns = [
            "Mobile Specification",
            "Entered Value"
        ]

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "The output represents the predicted price category "
        "from the trained dataset, not an exact market price."
    )


# ============================================================
# 11. FOOTER
# ============================================================

st.markdown("""
<div class="footer">

    MOBILE PRICE PREDICTION AI

    <br>

    POWERED BY SUPPORT VECTOR MACHINE

    <br>

    MACHINE LEARNING • CLASSIFICATION • ARTIFICIAL INTELLIGENCE

</div>
""", unsafe_allow_html=True)
