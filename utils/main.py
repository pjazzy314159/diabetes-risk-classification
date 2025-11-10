import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier  

db1 = pd.read_csv('data/db1.csv')
db2 = pd.read_csv('data/db2.csv')

X_train = db1.drop("Diabetes_binary", axis=1)
y_train = db1["Diabetes_binary"]

X_test = db2.drop("Diabetes_binary", axis=1)
y_test = db2["Diabetes_binary"]

pipeline = Pipeline([
    ('smote', SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=1)),
    ('scaler', StandardScaler()),
    ('model', XGBClassifier(
        random_state=42,
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=1,
        objective='multi:softmax',
        num_class=3
    ))
])

# === Train & Predict ===
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# === Save to file ===
db2_types = db2.copy()
db2_types['Predicted_Type'] = y_pred
db2_types.to_csv('data/db2_types_xgb.csv', index=False)
print("Saved predicted result to data/db2_types_xgb.csv")

# Reload for checking data
db2_types = pd.read_csv('data/db2_types_xgb.csv')
y_true = db2_types['Diabetes_binary']
y_pred = db2_types['Predicted_Type']

# Stats
print("\nPredicted type counts:\n", y_pred.value_counts())
print("\nTrue type counts:\n", y_true.value_counts())
print("\nUnique in y_true:", np.unique(y_true))
print("Unique in y_pred:", np.unique(y_pred))
print("\n=== Classification Report (Original vs Predicted) ===")
print(classification_report(y_true, y_pred, zero_division=0))
print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))

# Visualization
sns.set(style='whitegrid', context='talk')

# Scatter points of actual vs predicted
plt.figure(figsize=(8, 5))
plt.scatter(range(len(y_true)), y_true, alpha=0.6, s=10, label='Actual', color='royalblue')
plt.scatter(range(len(y_pred)), y_pred, alpha=0.4, s=10, label='Predicted', color='tomato')
plt.title('Predicted vs Actual Diabetes Types (Scatter Points)')
plt.xlabel('Sample Index')
plt.ylabel('Class (0=No, 1=Type1, 2=Type2)')
plt.legend()
plt.tight_layout()
plt.show()

# Bar chart of counts
plt.figure(figsize=(7, 4))
pred_counts = db2_types['Predicted_Type'].value_counts().sort_index()
sns.barplot(x=pred_counts.index, y=pred_counts.values, palette='viridis')
plt.title('Predicted Type Distribution (db2)')
plt.xlabel('Predicted Type')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Confusion matrices (binary vs multi)
y_true_bin = np.where(y_true >= 1, 1, 0)
y_pred_bin = np.where(y_pred >= 1, 1, 0)
cm_bin = confusion_matrix(y_true_bin, y_pred_bin)
cm_multi = confusion_matrix(y_true, y_pred)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Binary Confusion (0 vs 1/2)')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_xticklabels(['No', 'Diabetes'])
axes[0].set_yticklabels(['No', 'Diabetes'])

sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Multi-class Confusion (0/1/2)')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_xticklabels(['0', '1', '2'])
axes[1].set_yticklabels(['0', '1', '2'])
plt.tight_layout()
plt.show()
