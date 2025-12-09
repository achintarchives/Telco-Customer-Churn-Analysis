# Consolidated Python script for Telco Churn Project (Fixed dataset path issue placeholder)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load and clean data
# Replace the file path below with the actual correct path to your CSV
df = pd.read_csv(r"D:\Telco_Customer_Churn_Dataset .csv")
df.columns = df.columns.str.strip()
df = df.replace(r'^\s*$', pd.NA, regex=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# 2. Customer segmentation columns
df['TenureSegment'] = pd.cut(df['tenure'], bins=[0, 12, 36, 100], labels=['New', 'Mid', 'Loyal'], right=False)
df['SpendSegment'] = pd.cut(df['MonthlyCharges'], bins=[0, 35, 70, 150], labels=['Low', 'Medium', 'High'], right=False)

# 3. Prepare for modeling
df = df.drop('customerID', axis=1)
df_clean = pd.get_dummies(df, drop_first=True)
y = df_clean['Churn_Yes']
X = df_clean.drop('Churn_Yes', axis=1)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 6. Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Model Accuracy:", round(accuracy, 3))
print("\nClassification Report:\n", report)

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 7. Customer Lifetime Value (LTV)
avg_revenue = df['MonthlyCharges'].mean()
avg_lifetime = df['tenure'].mean()
LTV = avg_revenue * avg_lifetime
print("Customer Lifetime Value (LTV):", round(LTV, 2))

# 8. Identify high-value at-risk customers
df_model = df_clean.copy()
df_model['churn_prob'] = model.predict_proba(X)[:, 1]

# Align shapes between df_model and df for LTV calculation
df_model['LTV'] = df['MonthlyCharges'].reset_index(drop=True) * df['tenure'].reset_index(drop=True)

high_value_at_risk = df_model[
    (df_model['LTV'] > df_model['LTV'].median()) & (df_model['churn_prob'] > 0.6)
]

print("\nHigh-Value At-Risk Customers (Top 5):")
print(high_value_at_risk.head())
