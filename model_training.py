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

# ==============================
# Load Dataset
# ==============================
df = pd.read_csv("C:/Users/Mitali/OneDrive/Desktop/Python WorkSpace/emi_prediction/emi_prediction_dataset.csv")

# ==============================
# Data Cleaning
# ==============================
df.info()
df.isnull().sum()

# Fill missing values in 'education' with most frequent value
df['education'].fillna(df['education'].mode()[0], inplace=True)

# Convert 'bank_balance' to numeric, coercing errors
df['bank_balance'] = pd.to_numeric(df['bank_balance'], errors='coerce')

# Fill numeric columns with mean
numeric_cols = ['monthly_rent', 'credit_score', 'bank_balance', 'emergency_fund']
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

# Normalize gender values
df['gender'] = df['gender'].astype(str).str.strip().str.lower()
df['gender'] = df['gender'].apply(lambda x: 'Female' if x in ['female', 'f', 'female', 'FEMALE'] else 'Male')

# ==============================
# Encode Categorical Columns
# ==============================
le = LabelEncoder()
for col in df.select_dtypes(include=['object', 'category']).columns:
    df[col] = le.fit_transform(df[col].astype(str))

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
emi_corr = corr_matrix['emi_eligibility'].sort_values(ascending=False)

print("Correlation of all columns (including categorical) with EMI Eligibility:\n")
print(emi_corr)

# Visualize correlations
plt.figure(figsize=(8, 12))
sns.barplot(x=emi_corr.values, y=emi_corr.index, palette="coolwarm")
plt.title("Correlation of All Columns with EMI Eligibility")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Features")
plt.show()

# ==============================
# Group Analysis
# ==============================
print(df.groupby('gender')['emi_eligibility'].mean())
print(df.groupby('education')['emi_eligibility'].mean())

plt.figure(figsize=(6,4))
sns.countplot(x='education', hue='emi_eligibility', data=df)
plt.title("Education Level vs EMI Eligibility")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='gender', hue='emi_eligibility', data=df)
plt.title("Gender vs EMI Eligibility")
plt.show()

financial_cols = ['credit_score', 'bank_balance', 'monthly_salary',
                  'existing_loans', 'current_emi_amount', 'emergency_fund']
sns.heatmap(df[financial_cols + ['emi_eligibility']].corr(), annot=True, cmap='coolwarm')
plt.title("Financial Risk Factors and EMI Eligibility Correlation")
plt.show()

# ==============================
# Define Features & Target
# ==============================
X = df.drop(columns=['emi_eligibility'])
y = df['emi_eligibility']

print("Target variable counts before balancing:")
print(y.value_counts())

# Remove rare classes (if any)
value_counts = y.value_counts()
to_remove = value_counts[value_counts < 2].index
if len(to_remove) > 0:
    df = df[~df['emi_eligibility'].isin(to_remove)]
    X = df.drop(columns=['emi_eligibility'])
    y = df['emi_eligibility']
    print("\nAfter removing rare classes:")
    print(y.value_counts())

# ==============================
# Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# Initialize Models
# ==============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
}

# ==============================
# Model Training + Evaluation + MLflow Tracking
# ==============================
import mlflow
import mlflow.sklearn

# ✅ Explicitly set tracking URI (local by default)
mlflow.set_tracking_uri("file:///C:/Users/Mitali/OneDrive/Desktop/Python WorkSpace/emi_prediction/model_training")

# ✅ Create or set experiment
mlflow.set_experiment("EMI_Eligibility_Prediction")


results = []

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        # Train the model
        if name in ['Logistic Regression']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            if len(np.unique(y_test)) == 2:
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_prob = model.predict_proba(X_test_scaled)
            input_example = X_test_scaled[:1]  # for MLflow signature
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if len(np.unique(y_test)) == 2:
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.predict_proba(X_test)
            input_example = X_test[:1]  # for MLflow signature

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        average_param = 'binary' if len(np.unique(y_test)) == 2 else 'weighted'
        prec = precision_score(y_test, y_pred, average=average_param)
        rec = recall_score(y_test, y_pred, average=average_param)
        f1 = f1_score(y_test, y_pred, average=average_param)

        if len(np.unique(y_test)) == 2:
            roc = roc_auc_score(y_test, y_prob)
        else:
            from sklearn.preprocessing import LabelBinarizer
            lb = LabelBinarizer()
            y_test_bin = lb.fit_transform(y_test)
            roc = roc_auc_score(y_test_bin, y_prob, average='weighted')

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc)
        
        mlflow.sklearn.log_model(
        sk_model=model,
        name=name.replace(" ", "_") + "_model",
        registered_model_name=name.replace(" ", "_")
       )

        

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': roc
        })

        print(f"\n===== {name} =====")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC: {roc:.4f}")


# ==============================
# Results Summary
# ==============================
results_df = pd.DataFrame(results)
print("\n================ Summary of All Models ================\n")
print(results_df.sort_values(by='ROC-AUC', ascending=False).reset_index(drop=True))

best_model_name = results_df.sort_values(by='ROC-AUC', ascending=False).iloc[0]['Model']
print(f"\n🏆 Best Performing Model: {best_model_name}")
import joblib
import os

# ==============================
# Save Final Processed Dataset
# ==============================
final_dataset_path = "C:/Users/Mitali/OneDrive/Desktop/Python WorkSpace/emi_prediction/final_dataset.csv"
df.to_csv(final_dataset_path, index=False)
print(f"\n✅ Final cleaned dataset saved at: {final_dataset_path}")

# ==============================
# Save Best Performing Model
# ==============================
best_model = models[best_model_name]

# Refit model on the full training data for final saving
if best_model_name == "Logistic Regression":
    best_model.fit(X_train_scaled, y_train)
    model_save_path = "C:/Users/Mitali/OneDrive/Desktop/Python WorkSpace/emi_prediction/best_model_EMI_Eligibility_Prediction.pkl"
    joblib.dump((best_model, scaler), model_save_path)  # Save model + scaler
    print(f"✅ Best model ('{best_model_name}') and scaler saved at: {model_save_path}")
else:
    best_model.fit(X_train, y_train)
    model_save_path = "C:/Users/Mitali/OneDrive/Desktop/Python WorkSpace/emi_prediction/best_model_EMI_Eligibility_Prediction.pkl"
    joblib.dump(best_model, model_save_path)
    print(f"✅ Best model ('{best_model_name}') saved at: {model_save_path}")

