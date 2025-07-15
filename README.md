# credit-card-fraud-detection-xgboost

This project builds a machine learning pipeline to detect fraudulent credit card transactions using XGBoost. It includes exploratory data analysis (EDA), model selection (logistic regression, random forest, XGBoost), and hyperparameter tuning.

## Project Structure

notebooks/ # Jupyter notebooks for EDA and model selection
data/ # Dataset (Kaggle credit card fraud dataset)
models/ # Saved model (.pkl) for deployment
src/ # Core Python scripts (optional)


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
