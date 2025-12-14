# Credit Card Fraud Detection (Machine Learning)


## 📌 Overview
This project solves a real-world **credit card fraud detection** problem using classical machine learning techniques.
The dataset is **highly imbalanced**, making accuracy misleading and requiring advanced evaluation strategies.


## 🚀 Features
- Real Kaggle dataset (284K+ transactions)
- Severe class imbalance handling (SMOTE)
- Logistic Regression, Random Forest, XGBoost
- ROC-AUC & Precision-Recall evaluation
- Production-style ML pipeline
- Hyperparameter tuning with GridSearchCV


## 🧠 Why Accuracy is Misleading
If a model predicts every transaction as non-fraud, it achieves >99% accuracy while detecting **zero fraud cases**.
Hence, we focus on:
- ROC-AUC
- Precision
- Recall
- F1-score


## 📊 Models Used
- Logistic Regression (baseline)
- Random Forest (non-linear)
- XGBoost (high-performance)


## 🧪 How to Run
```bash
python src/train.py