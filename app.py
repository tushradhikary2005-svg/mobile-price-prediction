
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
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
# 2. CUSTOM CSS - MODERN DARK INTERFACE
# ==================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 5%, #182b50 0%, transparent 35%),
        radial-gradient(circle at 90% 20%, #291c50 0%, transparent 35%),
        #080d1b;
    color: #f1f5f9;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.hero {
    background: linear-gradient(120deg, #14294a, #30205a);
    border: 1px solid #35466f;
    border-radius: 24px;
    padding: 38px 35px;
    margin-bottom: 30px;
    box-shadow: 0 15px 50px rgba(0,0,0,0.25);
}

.hero-tag {
    color: #93c5fd;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.hero-title {
    color: white;
    font-size: 38px;
    font-weight: 800;
    margin: 12px 0;
    line-height: 1.2;
}

.hero-desc {
    color: #cbd5e1;
    font-size: 15px;
    line-height: 1.7;
}

.section-title {
    color: #f8fafc;
    font-size: 23px;
    font-weight: 750;
    margin-top: 20px;
    margin-bottom: 6px;
}

.section-subtitle {
    color: #94a3b8;
    font-size: 13px;
    margin-bottom: 20px;
}

.input-card {
    background: linear-gradient(145deg, #151f35, #10192b);
    border: 1px solid #293956;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 18px;
}

.card-heading {
    color: #93c5fd;
    font-size: 15px;
    font-weight: 750;
    margin-bottom: 12px;
}

div[data-testid="stNumberInput"] label {
    color: #dbeafe !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

div[data-testid="stNumberInput"] input {
    background-color: #0b1425 !important;
    color: white !important;
    border: 1px solid #354766 !important;
    border-radius: 10px !important;
}

div[data-testid="stNumberInput"] button {
    color: #93c5fd !important;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(100deg, #2563eb, #7c3aed);
    color: white !important;
    border: none;
    border-radius: 12px;
    min-height: 54px;
    font-size: 16px;
    font-weight: 750;
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.25);
    transition: 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(124, 58, 237, 0.4);
}

.result-card {
    background: linear-gradient(135deg, #102e42, #192c50, #30205a);
    border: 1px solid #4386d8;
    border-radius: 22px;
    padding: 35px 20px;
    text-align: center;
    margin-top: 22px;
    box-shadow: 0 15px 40px rgba(37,99,235,0.18);
}

.result-label {
    color: #bfdbfe;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.result-value {
    color: #67e8f9;
    font-size: 46px;
    font-weight: 850;
    margin: 15px 0;
    overflow-wrap: anywhere;
}

.result-note {
    color: #cbd5e1;
    font-size: 13px;
}

div[data-testid="stExpander"] {
    background: #111c30;
    border: 1px solid #293956;
    border-radius: 12px;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding-top: 30px;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. HERO SECTION
# ==================================================

st.markdown("""
<div class="hero">

    <div class="hero-tag">
        MACHINE LEARNING PROJECT
    </div>

    <div class="hero-title">
        Mobile Price<br>Prediction AI
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


# ==================================================
# 5. TRAIN SVM MODEL
# ==================================================

@st.cache_resource
def train_model():

    df = pd.read_csv(DATA_FILE)

    # Remove unnecessary unnamed columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Check required columns
    missing = [
        col for col in FEATURES + [TARGET]
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    # Remove missing values
    df = df.dropna(subset=FEATURES + [TARGET])

    # Separate features and target
    X = df[FEATURES]
    y = df[TARGET]

    if y.nunique() < 2:
        raise ValueError(
            "Dataset must contain at least two price classes."
        )

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Support Vector Machine model
    model = Pipeline([
        ("scaler", StandardScaler()),

        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced"
        ))
    ])

    # Train model
    model.fit(X_train, y_train)

    return model, df


# ==================================================
# 6. LOAD TRAINED MODEL
# ==================================================

try:

    with st.spinner("Preparing AI prediction model..."):
        model, df = train_model()

except FileNotFoundError:

    st.error(
        "Dataset not found! Upload mobile_price_dataset.csv "
        "to the same GitHub repository as app.py."
    )

    st.stop()

except Exception as e:

    st.error(f"Unable to prepare model: {e}")

    st.stop()


# ==================================================
# 7. INPUT LIMITS
# ==================================================

def get_limits(feature):

    values = pd.to_numeric(
        df[feature],
        errors="coerce"
    ).dropna()

    if values.empty:
        raise ValueError(f"No numeric values found in {feature}.")

    return (
        float(values.min()),
        float(values.max()),
        float(values.median())
    )


# ==================================================
# 8. MOBILE SPECIFICATION INPUT FORM
# ==================================================

st.markdown("""
<div class="section-title">
    ⚙️ Mobile Specifications
</div>

<div class="section-subtitle">
    Enter the hardware details to generate your prediction.
</div>
""", unsafe_allow_html=True)


with st.form("mobile_prediction_form"):

    col1, col2 = st.columns(2, gap="large")

    with col1:

        st.markdown("""
        <div class="input-card">
            <div class="card-heading">📸 CAMERA SYSTEM</div>
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
        <div class="input-card">
            <div class="card-heading">🧠 MEMORY</div>
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
        <div class="input-card">
            <div class="card-heading">💾 STORAGE</div>
        </div>
        """, unsafe_allow_html=True)

        storage_min, storage_max, storage_default = get_limits(
            "Storage_GB"
        )

        storage = st.number_input(
            "Internal Storage (GB)",
            min_value=storage_min,
            max_value=storage_max,
            value=storage_default
        )

        st.markdown("""
        <div class="input-card">
            <div class="card-heading">🔋 BATTERY</div>
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
        <div class="input-card">
            <div class="card-heading">📱 DISPLAY</div>
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
        "✦  PREDICT MOBILE PRICE RANGE",
        use_container_width=True
    )


# ==================================================
# 9. PREDICTION RESULT
# ==================================================

if submitted:

    # Prepare input data
    user_data = pd.DataFrame([{

        "Rear_Main_Camera_MP": rear_camera,

        "Rear_Camera_Count": rear_count,

        "Front_Camera_MP": front_camera,

        "RAM_GB": ram,

        "Storage_GB": storage,

        "Battery_mAh": battery,

        "Display_Size_inches": display

    }], columns=FEATURES)

    # Predict target variable
    with st.spinner("Analyzing mobile specifications..."):

        prediction = model.predict(user_data)[0]

    st.markdown("""
    <div class="section-title">
        ✨ Prediction Result
    </div>
    """, unsafe_allow_html=True)

    # Display result
    st.markdown(
        f"""
        <div class="result-card">

            <div class="result-label">
                PREDICTED PRICE RANGE (INR)
            </div>

            <div class="result-value">
                {prediction}
            </div>

            <div class="result-note">
                AI-generated classification result
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.success(
        "Your mobile phone specifications have been successfully analyzed."
    )

    # Display entered specifications
    with st.expander("📋 View Entered Specifications"):

        display_data = user_data.T.reset_index()

        display_data.columns = [
            "Mobile Specification",
            "Input Value"
        ]

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "Note: The output is a predicted price category, "
        "not an exact retail market price."
    )


# ==================================================
# 10. FOOTER
# ==================================================

st.markdown("""
<div class="footer">

    MOBILE PRICE PREDICTION AI<br><br>

    Powered by Support Vector Machine (SVM)<br>

    Machine Learning | Classification

</div>
""", unsafe_allow_html=True)
