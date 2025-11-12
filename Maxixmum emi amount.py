# ==============================
# Import Libraries
# ==============================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
import mlflow
import mlflow.sklearn


df = pd.read_csv("C:/Users/Mitali/OneDrive/Desktop/Python WorkSpace/emi_prediction/final_dataset.csv")
# ==============================
# Correlation Analysis
# ==============================
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", square=True)
plt.title("Correlation Matrix (Label Encoded)")
plt.show()

# One-hot encoding for full correlation
df_encoded = pd.get_dummies(df, drop_first=True)
corr_matrix = df_encoded.corr()
max_emi_amount_corr = corr_matrix['max_monthly_emi'].sort_values(ascending=False)

print("Correlation of all columns (including categorical) with max_monthly_emi:\n")
print(max_emi_amount_corr)

# Visualize correlations
plt.figure(figsize=(8, 12))
sns.barplot(x=max_emi_amount_corr.values, y=max_emi_amount_corr.index, palette="coolwarm")
plt.title("Correlation of All Columns with max_emi_amount")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Features")
plt.show()

# ==============================
# Regression Model Training for Max Monthly EMI Prediction
# ==============================
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib

# Define features & target
X = df.drop(columns=['max_monthly_emi'])
y = df['max_monthly_emi']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize regression models
regressors = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42),
    "XGBoost Regressor": XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=200, random_state=42)
}

# Function to compute metrics
def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mae, r2, mape

results_reg = []

print("\n================ Regression Model Evaluation ================\n")

# Train and evaluate models
for name, model in regressors.items():
    print(f"\n===== {name} =====")

    # Use scaled data only for Linear Regression
    if name == "Linear Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    rmse, mae, r2, mape = evaluate_model(y_test, y_pred)

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")

    results_reg.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
    })

# ==============================
# Compare All Regression Models
# ==============================
results_reg_df = pd.DataFrame(results_reg)
print("\n================ Summary of Regression Models ================\n")
print(results_reg_df.sort_values(by='R²', ascending=False).reset_index(drop=True))

best_regressor_name = results_reg_df.sort_values(by='R²', ascending=False).iloc[0]['Model']
print(f"\n🏆 Best Performing Regressor: {best_regressor_name}")

# ==============================
# Save Best Regressor Model
# ==============================
best_regressor = regressors[best_regressor_name]
if best_regressor_name == "Linear Regression":
    best_regressor.fit(X_train_scaled, y_train)
    model_save_path = "C:/Users/Mitali/OneDrive/Desktop/Python WorkSpace/emi_prediction/best_regressor_model.pkl"
    joblib.dump((best_regressor, scaler), model_save_path)
else:
    best_regressor.fit(X_train, y_train)
    model_save_path = "C:/Users/Mitali/OneDrive/Desktop/Python WorkSpace/emi_prediction/best_regressor_model.pkl"
    joblib.dump(best_regressor, model_save_path)

print(f"✅ Best Regressor Model ('{best_regressor_name}') saved at: {model_save_path}")

# ==============================
# Save Final Regression Dataset
# ==============================
final_regression_dataset_path = "C:/Users/Mitali/OneDrive/Desktop/Python WorkSpace/emi_prediction/final_regression_dataset.csv"
df.to_csv(final_regression_dataset_path, index=False)
print(f"✅ Final dataset for regression saved at: {final_regression_dataset_path}")

# ==============================
# Optional: MLflow Logging
# ==============================
mlflow.set_experiment("Max_Monthly_EMI_Prediction")

with mlflow.start_run(run_name=f"Best_{best_regressor_name}"):
    mlflow.log_params({"target": "max_monthly_emi"})
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("MAPE", mape)
    mlflow.sklearn.log_model(best_regressor, best_regressor_name.replace(" ", "_") + "_best_regressor")
    mlflow.log_artifact(final_regression_dataset_path)

print("📦 Best regression model and dataset logged to MLflow.")
