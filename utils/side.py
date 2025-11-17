import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
import joblib


db1 = pd.read_csv('data/db1.csv')
db2 = pd.read_csv('data/db2.csv')

y_train = db1['Diabetes_binary'].replace({2: 1})
X_train = db1.drop('Diabetes_binary', axis=1)

X_test = db2.drop('Diabetes_binary', axis=1)
y_test = db2['Diabetes_binary']


# Without ADASYN
pipeline_tune = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    ))
])

param_grid = {
    'model__max_depth': [4, 6],
    'model__learning_rate': [0.01, 0.05],
    'model__n_estimators': [200, 400]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=pipeline_tune,
    param_distributions=param_grid,
    n_iter=8,
    cv=cv,
    scoring='f1',
    verbose=1,
    n_jobs=1
)

search.fit(X_train, y_train)

print("\nBest Params:", search.best_params_)
print("Best CV Score:", search.best_score_)

best_params = search.best_params_

#With ADASYN
pipeline = Pipeline([
    ('adasyn', ADASYN(n_neighbors=3, random_state=42)),
    ('scaler', StandardScaler()),
    ('model', XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        max_depth=best_params['model__max_depth'],
        learning_rate=best_params['model__learning_rate'],
        n_estimators=best_params['model__n_estimators']
    ))
])

pipeline.fit(X_train, y_train)
y_pred_stage2 = pipeline.predict(X_test)

db2_types_compare = db2.copy()
db2_types_compare['Predicted_Type'] = y_pred_stage2

# db2_types_compare.to_csv('data/db2_types_compare_final.csv', index=False)


print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred_stage2, zero_division=0))

cm = confusion_matrix(y_test, y_pred_stage2)
print("\nConfusion Matrix:\n", cm)

sns.set(style='whitegrid', context='talk')

plt.figure(figsize=(8, 4))
bar_width = 0.35
x = np.arange(2)
actual_counts = pd.Series(y_test).value_counts().sort_index()
pred_counts = pd.Series(y_pred_stage2).value_counts().sort_index()

plt.bar(x - bar_width/2, actual_counts, width=bar_width, label='Actual', color='skyblue')
plt.bar(x + bar_width/2, pred_counts, width=bar_width, label='Predicted', color='salmon')

plt.title('Actual vs Predicted Diabetes Classes (0 vs 1)')
plt.xlabel('0 = No, 1 = Diabetes')
plt.ylabel('Count')
plt.xticks(x, ['0', '1'])
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks([0.5, 1.5], ['0', '1'])
plt.yticks([0.5, 1.5], ['0', '1'])
plt.tight_layout()
plt.show()
