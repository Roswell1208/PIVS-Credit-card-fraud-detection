import streamlit as st
import pandas as pd
from PIL import Image

# === PAGE SETUP ===
st.set_page_config(page_title="ML Model Explorer", layout="centered")

# === PAGE SELECTION ===
page = st.sidebar.radio("Choose a page", ["Dataset Visualization", "Model Evaluation"])

# === PAGE 1: DATASET VISUALIZATION ===
if page == "Dataset Visualization":
    st.title("Dataset Overview")

    # Section 1 — Aperçu du dataset
    st.subheader("Sample of the Dataset")
    try:
        df = pd.read_csv("DataSet/creditcard_sample.csv")
        st.dataframe(df)
    except FileNotFoundError:
        st.error("head_sample.csv not found in /data")

    # Section 2 — Distribution visuelle
    st.subheader("Class Distribution")

    dataset_choice = st.selectbox("Select dataset type", ["Original", "Undersampled", "Oversampled"])
    image_map = {
        "Original": "class_distributions/original.png",
        "Undersampled": "class_distributions/undersampled.png",
        "Oversampled": "class_distributions/oversampled.png"
    }

    try:
        image = Image.open(image_map[dataset_choice])
        st.image(image, caption=f"{dataset_choice} class distribution", use_container_width=True)
    except FileNotFoundError:
        st.error(f"{image_map[dataset_choice]} not found.")

# === PAGE 2: MODEL EVALUATION ===
elif page == "Model Evaluation":
    st.title("Model Confusion Matrices")

    # Dictionnaire des modèles et chemins des images
    models = {
        "Random Forest": {
            "Original": "confusion_matrices/rf_original.png",
            "Undersampling": "confusion_matrices/rf_undersampling.png",
            "Oversampling": "confusion_matrices/rf_oversampling.png"
        },
        "K-Nearest Neighbors": {
            "Original": "confusion_matrices/knn_original.png",
            "Undersampling": "confusion_matrices/knn_undersampling.png",
            "Oversampling": "confusion_matrices/knn_oversampling.png"
        },
        "XGBoost": {
            "Original": "confusion_matrices/xgb_original.png",
            "Undersampling": "confusion_matrices/xgb_undersampling.png",
            "Oversampling": "confusion_matrices/xgb_oversampling.png"
        },
        "Logistic Regression": {
            "Original": "confusion_matrices/logreg_original.png",
            "Undersampling": "confusion_matrices/logreg_undersampling.png",
            "Oversampling": "confusion_matrices/logreg_oversampling.png"
        }
    }

    # Observations par modèle
    observations = {
        "Random Forest": """
**Observations for Random Forest:**  
Lorem ipsum dolor sit amet, consectetur adipiscing elit.  
""",
        "K-Nearest Neighbors": """
**Observations for K-Nearest Neighbors:**  
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.  
""",
        "XGBoost": """
**Observations for XGBoost:**  
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore.  
""",
        "Logistic Regression": """
**Observations for Logistic Regression:**  
Excepteur sint occaecat cupidatat non proident.  
"""
    }

    selected_model = st.sidebar.selectbox("Select a model", list(models.keys()))

    st.header(f"{selected_model} - Confusion Matrices")

    for dataset_type, image_path in models[selected_model].items():
        st.subheader(f"{dataset_type} Dataset")
        try:
            image = Image.open(image_path)
            st.image(image, caption=f"{selected_model} - {dataset_type}", use_container_width=True)
        except FileNotFoundError:
            st.warning(f"Image not found: {image_path}")

    st.markdown("### Observations")
    st.markdown(observations[selected_model])
