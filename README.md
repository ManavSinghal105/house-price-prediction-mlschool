%%writefile README.md
# 🏡 House Price Prediction - Supervised Learning (Amazon ML Summer School)

This project is part of **Amazon ML Summer School 2025**, under the first topic: **Supervised Learning**.  
It demonstrates how to train and compare regression models on housing price prediction.

## 📌 Models Used
- Linear Regression
- Decision Tree Regressor
- Random Forest (with GridSearch hyperparameter tuning)

## ⚙️ Features
- Compare model performance (MAE, RMSE, R² score).
- Visualize feature importance for tree-based models.
- Interactive **Streamlit app** to test models.
- Trained on `California Housing` dataset (from scikit-learn).

## 🚀 Running the Streamlit App
```bash
pip install -r requirements.txt
streamlit run app.py
