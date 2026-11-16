
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Mobile Price Prediction",
    page_icon="📱",
    layout="centered"
)

st.title("📱 Mobile Price Range Prediction")
st.write(
    "Enter the mobile phone specifications below "
    "to predict its price range using Machine Learning."
)

st.markdown("---")

# -----------------------------------
# DATASET CONFIGURATION
# -----------------------------------

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

# -----------------------------------
# LOAD DATASET AND TRAIN MODEL
# -----------------------------------

@st.cache_resource
def train_model():

    df = pd.read_csv(DATA_FILE)

    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    missing = [col for col in FEATURES + [TARGET] if col not in df.columns]

    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    df = df.dropna(subset=FEATURES + [TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    if y.nunique() < 2:
        raise ValueError("Dataset must contain at least two price classes.")

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

    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, accuracy, df


try:
    model, accuracy, df = train_model()

except FileNotFoundError:
    st.error(
        "Dataset not found! Please upload "
        "mobile_price_dataset.csv to your GitHub repository."
    )
    st.stop()

except Exception as e:
    st.error(f"Model training error: {e}")
    st.stop()

# -----------------------------------
# MODEL INFORMATION
# -----------------------------------

st.success("SVM Model Successfully Loaded!")

st.metric(
    label="Model Test Accuracy",
    value=f"{accuracy * 100:.2f}%"
)

st.markdown("---")

# -----------------------------------
# USER INPUT SECTION
# -----------------------------------

st.header("Enter Mobile Specifications")

st.write("Provide the following details:")

col1, col2 = st.columns(2)

# Get dataset limits for sensible input ranges
def limits(feature):
    values = pd.to_numeric(df[feature], errors="coerce").dropna()
    return float(values.min()), float(values.max()), float(values.median())

with col1:

    rear_camera_min, rear_camera_max, rear_camera_default = limits(
        "Rear_Main_Camera_MP"
    )

    rear_camera = st.number_input(
        "Rear Main Camera (MP)",
        min_value=rear_camera_min,
        max_value=rear_camera_max,
        value=rear_camera_default
    )

    count_min, count_max, count_default = limits(
        "Rear_Camera_Count"
    )

    rear_count = st.number_input(
        "Number of Rear Cameras",
        min_value=int(count_min),
        max_value=int(count_max),
        value=int(count_default),
        step=1
    )

    front_min, front_max, front_default = limits(
        "Front_Camera_MP"
    )

    front_camera = st.number_input(
        "Front Camera (MP)",
        min_value=front_min,
        max_value=front_max,
        value=front_default
    )

    ram_min, ram_max, ram_default = limits("RAM_GB")

    ram = st.number_input(
        "RAM (GB)",
        min_value=ram_min,
        max_value=ram_max,
        value=ram_default
    )

with col2:

    storage_min, storage_max, storage_default = limits("Storage_GB")

    storage = st.number_input(
        "Storage (GB)",
        min_value=storage_min,
        max_value=storage_max,
        value=storage_default
    )

    battery_min, battery_max, battery_default = limits("Battery_mAh")

    battery = st.number_input(
        "Battery Capacity (mAh)",
        min_value=battery_min,
        max_value=battery_max,
        value=battery_default
    )

    display_min, display_max, display_default = limits(
        "Display_Size_inches"
    )

    display = st.number_input(
        "Display Size (inches)",
        min_value=display_min,
        max_value=display_max,
        value=display_default
    )

# -----------------------------------
# PREDICTION BUTTON
# -----------------------------------

st.markdown("---")

if st.button("🔍 Predict Price Range", use_container_width=True):

    user_data = pd.DataFrame([{
        "Rear_Main_Camera_MP": rear_camera,
        "Rear_Camera_Count": rear_count,
        "Front_Camera_MP": front_camera,
        "RAM_GB": ram,
        "Storage_GB": storage,
        "Battery_mAh": battery,
        "Display_Size_inches": display
    }], columns=FEATURES)

    prediction = model.predict(user_data)[0]

    st.markdown("---")

    st.header("Prediction Result")

    st.success("Prediction Completed Successfully!")

    st.subheader("Predicted Target Variable")

    st.markdown(
        f"""
        <div style="
            background-color:#123524;
            padding:25px;
            border-radius:12px;
            text-align:center;
            border:1px solid #239b56;
        ">
            <h3 style="color:#ffffff;">Price Range (INR)</h3>
            <h1 style="color:#4ade80;">{prediction}</h1>
            <p style="color:#ffffff;">Predicted Class</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    st.info(
        "The displayed value is the predicted target class "
        "from Price_Range_INR. It is a price category, "
        "not an exact mobile phone market price."
    )

    with st.expander("View Entered Mobile Specifications"):
        st.dataframe(user_data, use_container_width=True)

# -----------------------------------
# FOOTER
# -----------------------------------

st.markdown("---")

st.caption(
    "Mobile Price Range Prediction | "
    "Machine Learning Classification | SVM Algorithm"
)
