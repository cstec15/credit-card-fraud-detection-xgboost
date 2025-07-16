# Credit Card Fraud Detection with XGBoost

This repository contains a machine learning pipeline for detecting fraudulent credit card transactions using the popular Kaggle Credit Card Fraud Detection dataset. The goal is to identify fraudulent transactions (which represent only 0.17% of all transactions) with high recall and precision and maximize AUPRC.

This is an initial version of the project. Iterative improvements in progress.

## Project Structure

```
credit-card-fraud-detection-xgboost/
├── data/                  # (not included) Dataset folder, see Kaggle link in README
├── models/                # Saved models (pickled)
│   └── final_model.pkl    # XGBoost model with optimized threshold
├── notebooks/             # Jupyter notebooks for EDA & modeling
│   ├── Data Ingestion, Learning, and Exploration.ipynb
│   ├── Logistic Regression.ipynb
│   ├── Random Forest.ipynb
│   ├── XG Boost.ipynb
│   └── Final Pipeline.ipynb
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

```
## Dataset

This project uses the Kaggle Credit Card Fraud Detection dataset, containing 284,807 transactions where only 0.17% are fraudulent:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Due to size constraints, the dataset is not included in this repository.
Download it from Kaggle and place it in the data/ folder:
```bash 
data/creditcard.csv
```


## Final Model Performance

| Metric        | Score |
|---------------|-------|
| AUPRC         | 0.843 |
| Precision      | 0.81  |
| Recall         | 0.84  |

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Load and use the final model:
```python
import pickle

# Load saved model
with open('models/final_model.pkl', 'rb') as f:
    final_model = pickle.load(f)

# Make predictions
preds = final_model.predict(new_data)
