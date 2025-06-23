# 💳 Credit Card Fraud Detection

This project focuses on building and evaluating several machine learning models to detect fraudulent transactions in credit card usage. The key challenge addressed is the **severe class imbalance**, as fraud cases are extremely rare compared to legitimate transactions.


## 📁 Dataset

The dataset used is a well-known public dataset available on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It contains real transactions made by European cardholders in September 2013.

- **Total records**: 284,807 transactions  
- **Fraud cases**: 492  
- **Class ratio**: ~0.17% fraud

> **Note:** Due to confidentiality, original features are anonymized as V1 to V28, along with `Amount` and `Time`.


## ⚙️ Project Structure

```project
├── DataSet/
| └── original-dataset/test/X.csv
| └── original-dataset/test/y.csv
| └── original-dataset/train/X.csv
| └── original-dataset/train/y.csv
| └── oversampled-dataset/...
| └── undersampled-dataset/...
| └── creditcard_sample.csv
│ └── creditcard.csv (Download the dataset from Kaggle)
| └── dataset-cleaning.ipynb
├── class_distributions/
│ └── original.png / undersampled.png / oversampled.png
├── confusion_matrices/
│ └── [model_dataset].png
├── results/
│ └── f1_score_plot.png / recall_precision_plot.png / roc_auc_vs_f1_plot.png
│ └── fraud_detection_model_results.csv
│ └── results.ipynb
├── KNN
│ └── pipeline.ipynb
│ └── grid_search_knn_dataset.pkl
├── LogisticRegression
│ └── pipeline.ipynb
│ └── grid_search_lr_dataset.pkl
├── RandomForest
│ └── pipeline.ipynb
│ └── grid_search_rf_dataset.pkl
├── SVM
│ └── pipeline.ipynb
│ └── grid_search_svm_dataset.pkl
├── XGBoost
│ └── pipeline.ipynb
│ └── grid_search_xgb_dataset.pkl
├── streamlit-app.py
└── README.md
```


## 📊 Machine Learning Models

Five classification models were tested:

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

Each model was evaluated on three different versions of the dataset:

1. **Original dataset** (highly imbalanced)
2. **Undersampled** (balanced by reducing majority class)
3. **Oversampled** (balanced by duplicating minority class)


## ✅ Evaluation Metrics

Each model was assessed using:

- **F1-Score** (for Class 1)
- **Recall** and **Precision**
- **ROC AUC**
- **Confusion Matrix**

Special attention was paid to the trade-off between precision and recall, which is critical in fraud detection tasks where false positives and false negatives have different business implications.


## 📈 Results Summary

- **XGBoost on the original dataset** consistently performed best across all metrics, achieving a strong balance between fraud detection and minimizing false positives.
- Resampling techniques significantly influenced model behavior, often boosting recall but hurting precision.
- Simpler models (like Logistic Regression and KNN) were less robust to oversampling/undersampling variations.


## 🧪 Streamlit App

A simple interactive **Streamlit** application is included to explore:

- Dataset class distributions  
- Model confusion matrices  
- Final comparison plots and conclusions

To run it locally:
```bash
streamlit run streamlit-app.py
```


## 📌 Final Thoughts

This project highlights the importance of both model selection and data preprocessing when dealing with highly imbalanced datasets. Fraud detection systems must optimize not only for accuracy but also for practical, domain-specific costs related to prediction errors.


## 🔗 Dataset Source

👉 Kaggle Dataset – [Link Here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)