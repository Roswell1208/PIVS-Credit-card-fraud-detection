import streamlit as st
import pandas as pd
from PIL import Image

# === PAGE SETUP ===
st.set_page_config(page_title="ML Model Explorer", layout="centered")

# === PAGE SELECTION ===
page = st.sidebar.radio("Choose a page", ["Dataset Visualization", "Model Evaluation", "Final Analysis"])

# === PAGE 1: DATASET VISUALIZATION ===
if page == "Dataset Visualization":
    st.title("Dataset Overview")

    # Section 1 — Dataset preview
    st.subheader("Sample of the Dataset")
    try:
        df = pd.read_csv("DataSet/creditcard_sample.csv")
        st.dataframe(df)
    except FileNotFoundError:
        st.error("head_sample.csv not found in /data")

    # Section 2 — Class Distribution
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
        },
        "Support Vector Machine": {
            "Original": "confusion_matrices/svm_original.png",
            "Undersampling": "confusion_matrices/svm_undersampling.png",
            "Oversampling": "confusion_matrices/svm_oversampling.png"
        }
    }

    # Observations
    observations = {
        "Random Forest": """
**Observations for Random Forest:**  
Random Forest performs well on the original data, showing reliable classification. However, it reacts poorly to resampling, especially oversampling, where performance deteriorates significantly, suggesting sensitivity to class distribution shifts.
""",
        "K-Nearest Neighbors": """
**Observations for K-Nearest Neighbors:**  
KNN shows a strong tendency to favor recall when class imbalance is addressed, but suffers from drastic drops in precision and F1-score, particularly with oversampling, making it a risky choice for fraud detection where false positives are costly.
""",
        "XGBoost": """
**Observations for XGBoost:**  
XGBoost demonstrates robustness across different dataset versions. It performs best on the original and oversampled data, showing a strong ability to detect fraud while keeping false positives relatively low—making it a reliable model in this context.
""",
        "Logistic Regression": """
**Observations for Logistic Regression:**  
While logistic regression handles the original dataset reasonably well, its performance collapses on resampled datasets due to severe precision loss, showing that it struggles to maintain meaningful decision boundaries under class balancing techniques.
""",
        "Support Vector Machine": """
**Observations for SVM:**  
SVM struggles significantly with class imbalance. It either overpredicts the minority class with poor precision or underdetects it with low recall. Performance is especially unstable on oversampled data, making it unreliable for fraud detection in this setup.
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

# === PAGE 3: FINAL ANALYSIS ===
elif page == "Final Analysis":
    st.title("Final Model Analysis & Insights")

    st.subheader("📊 Global Performance Metrics")

    plots = {
        "F1-Score by Model and Dataset": "results/f1_score_plot.png",
        "Recall and Precision by Model": "results/recall_precision_plot.png",
        "ROC AUC vs F1-Score": "results/roc_auc_vs_f1_plot.png"
    }

    for title, plot_path in plots.items():
        st.markdown(f"#### {title}")
        try:
            image = Image.open(plot_path)
            st.image(image, caption=title, use_container_width=True)
        except FileNotFoundError:
            st.warning(f"Image not found: {plot_path}")

    st.subheader("F1-Score Trends")
    st.markdown("""
    XGBoost consistently achieves the highest F1-Score on the original and oversampled datasets, indicating a strong balance between precision and recall.

    KNN, Logistic Regression, and Random Forest also perform relatively well on the original dataset, with F1-Scores above 0.70.

    Undersampling severely impacts precision across nearly all models, leading to extremely low F1-Scores despite high recall.

    Oversampling generally improves recall but often degrades precision dramatically—especially for simpler models like Logistic Regression and KNN.
    """)

    st.subheader("Recall vs. Precision")
    st.markdown("""
    All models on undersampled data achieve very high recall, but this comes at the expense of extremely poor precision, resulting in numerous false positives.

    Original datasets strike a better trade-off: reasonable recall with high precision, especially for XGBoost, Random Forest, and Logistic Regression.

    Oversampled datasets show unstable behavior: some models (e.g., XGBoost) handle it well, while others (e.g., SVM, Logistic Regression) suffer from very poor precision.)
    """)

    st.subheader("ROC AUC vs. F1-Score")
    st.markdown("""
    High ROC AUC does not always correlate with a high F1-Score, especially in imbalanced datasets.

    For example, some undersampled models reach ROC AUCs near 0.99 but still have F1-Scores close to 0.05–0.10, highlighting the limitations of relying solely on ROC AUC in imbalanced settings.
    """)

    st.subheader("Conclusion")
    st.markdown("""
    Among all tested models and sampling strategies, XGBoost on the original dataset delivers the most balanced and effective performance, achieving high precision, strong recall, and the best F1-Score overall. While undersampling increases sensitivity (recall), it does so at the cost of excessive false positives, making it impractical for real-world fraud detection. Oversampling can help in some cases (e.g., XGBoost), but its success depends heavily on the model's capacity to generalize. Overall, the results emphasize the importance of model choice and careful handling of class imbalance in fraud detection tasks.
    """)