# Credit Card Fraud Detection with XGBoost

This repository contains a machine learning pipeline for detecting fraudulent credit card transactions using the popular Kaggle Credit Card Fraud Detection dataset. The goal is to identify fraudulent transactions (which represent only 0.17% of all transactions) with high recall and precision and maximize AUPRC.

## Project Structure

credit-card-fraud-detection-xgboost/
│
├── data/                   # (not included) Dataset folder, see below
├── models/                 # Saved models (pickled)
├── notebooks/              # Jupyter notebooks for EDA & modeling
│   ├── 1_eda.ipynb
│   ├── 2_logistic_regression.ipynb
│   ├── 3_random_forest.ipynb
│   ├── 4_xgboost.ipynb
├── final_model.pkl         # XGBoost model with optimized threshold
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

## Dataset

The dataset is not included in this repository due to size constraints. You can download it directly from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Once downloaded, place the CSV file(s) in a data/ directory at the root of this project:

## Final Model Performance

| Metric        | Score |
|---------------|-------|
| AUPRC         | 0.843 |
| Precision      | 0.81  |
| Recall         | 0.84  |

## Requirements

Install dependencies:
pip install -r requirements.txt


## Usage

Load and use the final model:
```python
import pickle

# Load saved model
with open('models/final_model.pkl', 'rb') as f:
    final_model = pickle.load(f)

# Make predictions
preds = final_model.predict(new_data)
