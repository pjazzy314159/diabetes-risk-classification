import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier

db1 = pd.read_csv('data/db1.csv')
db2 = pd.read_csv('data/db2.csv')

y_train = db1['Diabetes_binary'].replace({2: 1})
X_train = db1.drop('Diabetes_binary', axis=1)

X_test = db2.drop('Diabetes_binary', axis=1)
y_test = db2['Diabetes_binary']

svm_classifier = SVC(kernel = 'linear')
kf = KFold(n_splits=10, shuffle= True, random_state=42)

cross_val_res = cross_val_score(svm_classifier, X_train, y_train, cv=kf)

print('CVR: ')
for i, res in enumerate(cross_val_res,1):
    print(f"Fold {i}: {res}%")
    
print("ACC:", cross_val_res.mean())

pipeline_stage2 = Pipeline([
    ('adasyn', ADASYN(n_neighbors=3,random_state = 42)),
    ('scaler', StandardScaler()),
    ('model', XGBClassifier(
        random_state=42,
        n_estimators=400,
        learning_rate=0.02,
        scale_pos_weight=1))
])

pipeline_stage2.fit(X_train, y_train)
y_pred_stage2 = pipeline_stage2.predict(X_test)

db2_types_compare = db2.copy()
db2_types_compare['Predicted_Type'] = y_pred_stage2
db2_types_compare.to_csv('data/db2_types_compare.csv', index=False)
print("Saved: data/db2_types_compare.csv")


print("\nClassification Report:")
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
