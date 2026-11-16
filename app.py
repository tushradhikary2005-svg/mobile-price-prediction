
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    cohen_kappa_score, matthews_corrcoef
)

st.set_page_config(
    page_title="Mobile Price Prediction",
    page_icon="📱",
    layout="wide"
)

st.title("📱 Mobile Price Range Prediction")
st.write("A machine learning classification project using Support Vector Machine (SVM).")

# Dataset and features
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

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.dropna(subset=FEATURES + [TARGET])
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error(
        f"Dataset not found. Upload '{DATA_FILE}' "
        "to the same GitHub repository as app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

# Check required columns
missing = [c for c in FEATURES + [TARGET] if c not in df.columns]

if missing:
    st.error(f"Missing columns in dataset: {missing}")
    st.stop()

X = df[FEATURES]
y = df[TARGET]

if y.nunique() < 2:
    st.error("The target column must contain at least two classes.")
    st.stop()

# Train-test split
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
except ValueError:
    st.error(
        "The dataset does not have enough examples in every class "
        "for a stratified 80/20 split. Add more data for each class."
    )
    st.stop()

# Main SVM model
svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced"
    ))
])

svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# Navigation
page = st.sidebar.selectbox(
    "Choose a page",
    ["Project Overview", "Model Performance", "Predict Mobile Price Range"]
)

# PAGE 1: Overview
if page == "Project Overview":

    st.header("Project Overview")

    st.write("""
    This project predicts the price range of a mobile phone
    using its hardware specifications.

    The Support Vector Machine (SVM) algorithm is used as the
    main classification model.
    """)

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Records", len(df))
    c2.metric("Input Features", len(FEATURES))
    c3.metric("Price Classes", y.nunique())

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Price Range Distribution")
    fig, ax = plt.subplots()
    df[TARGET].value_counts().sort_index().plot(
        kind="bar", ax=ax
    )
    ax.set_xlabel("Price Range")
    ax.set_ylabel("Number of Mobile Phones")
    ax.set_title("Distribution of Price Classes")
    st.pyplot(fig)

# PAGE 2: Performance
elif page == "Model Performance":

    st.header("SVM Model Performance")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    sensitivity = recall_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{accuracy * 100:.2f}%")
    c2.metric("Precision", f"{precision * 100:.2f}%")
    c3.metric("Sensitivity", f"{sensitivity * 100:.2f}%")

    c4, c5, c6 = st.columns(3)
    c4.metric("F1 Score", f"{f1 * 100:.2f}%")
    c5.metric("Cohen's Kappa", f"{kappa:.4f}")
    c6.metric("MCC", f"{mcc:.4f}")

    st.subheader("Confusion Matrix")

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0
    ) * 100

    annotations = np.empty_like(cm, dtype=object)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = (
                f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"
            )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        cm_percent,
        annot=annotations,
        fmt="",
        cmap="Reds",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=100,
        ax=ax
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title("SVM Confusion Matrix")
    st.pyplot(fig)

    st.subheader("Classification Report")
    report = classification_report(
        y_test, y_pred, zero_division=0, output_dict=True
    )
    st.dataframe(pd.DataFrame(report).transpose())

    # Model comparison
    st.subheader("Comparison with Other Models")

    if st.button("Run Model Comparison"):
        models = {
            "SVM": Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced"
                ))
            ]),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, random_state=42
            ),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=200, random_state=42
            ),
            "KNN": Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=5))
            ]),
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=2000, class_weight="balanced"
                ))
            ]),
            "Gradient Boosting": GradientBoostingClassifier(
                random_state=42
            )
        }

        results = []

        for name, model in models.items():
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)

            results.append({
                "Model": name,
                "Accuracy (%)": round(
                    accuracy_score(y_test, prediction) * 100, 2
                )
            })

        result_df = pd.DataFrame(results).sort_values(
            by="Accuracy (%)", ascending=False
        )

        st.dataframe(result_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(9, 4))
        sns.barplot(
            data=result_df,
            x="Accuracy (%)",
            y="Model",
            ax=ax
        )
        ax.set_title("Model Accuracy Comparison")
        st.pyplot(fig)

# PAGE 3: Prediction
elif page == "Predict Mobile Price Range":

    st.header("Predict a Mobile Phone's Price Range")
    st.write("Enter the mobile phone specifications below.")

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        values = {}

        for i, feature in enumerate(FEATURES):
            series = pd.to_numeric(df[feature], errors="coerce").dropna()

            if series.empty:
                st.error(f"No valid numeric values found for {feature}.")
                st.stop()

            default = float(series.median())
            minimum = float(series.min())
            maximum = float(series.max())

            with (col1 if i % 2 == 0 else col2):
                values[feature] = st.number_input(
                    feature.replace("_", " "),
                    min_value=minimum,
                    max_value=maximum,
                    value=default
                )

        submitted = st.form_submit_button("Predict Price Range")

    if submitted:
        input_df = pd.DataFrame([values], columns=FEATURES)
        prediction = svm_model.predict(input_df)[0]

        st.success(f"Predicted Price Range: {prediction}")

        st.info(
            "This result is the predicted class from the dataset. "
            "It is not an exact market price."
        )

st.sidebar.markdown("---")
st.sidebar.caption("Mobile Price Range Prediction | SVM Classification")
