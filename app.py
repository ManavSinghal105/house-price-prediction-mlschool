import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------
# Load Data
# ---------------------------
st.title("🏡 House Price Prediction (Supervised Learning Mini-Project)")

data = fetch_california_housing(as_frame=True)
df = data.frame

st.subheader("Dataset Preview")
st.write(df.head())

# ---------------------------
# Train/Test Split
# ---------------------------
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# Sidebar for Model Selection
# ---------------------------
st.sidebar.title("Choose a Model")
model_choice = st.sidebar.selectbox("Model", ["Linear Regression", "Decision Tree", "Random Forest (Tuned)"])

# ---------------------------
# Train Models
# ---------------------------
if model_choice == "Linear Regression":
    model = LinearRegression()
elif model_choice == "Decision Tree":
    model = DecisionTreeRegressor(random_state=42)
elif model_choice == "Random Forest (Tuned)":
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X_train, y_train)
    st.write("Best Parameters:", grid.best_params_)
    model = grid.best_estimator_

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---------------------------
# Show Metrics
# ---------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write(f"**MAE:** {mae:.4f}")
st.write(f"**RMSE:** {rmse:.4f}")
st.write(f"**R² Score:** {r2:.4f}")

# ---------------------------
# Feature Importance (only for Tree/Forest)
# ---------------------------
if model_choice in ["Decision Tree", "Random Forest (Tuned)"]:
    st.subheader("Feature Importance")
    importances = model.feature_importances_
    fig, ax = plt.subplots()
    ax.bar(X.columns, importances)
    ax.set_xticks(range(len(X.columns)))
    ax.set_xticklabels(X.columns, rotation=90)
    st.pyplot(fig)
