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
    ('smote', SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=2)),
    ('scaler', StandardScaler()),
    ('model', XGBClassifier(
        random_state=42,
        n_estimators=400,
        learning_rate=0.05,
        objective='multi:softmax',
        num_class = 3))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

db2_types = db2.copy()
db2_types['Predicted_Type'] = y_pred
db2_types.to_csv('data/db2_types_xgb.csv', index=False)
print("Saved predicted result to data/db2_types_xgb.csv")

db2_types = pd.read_csv('data/db2_types_xgb.csv')
print(db2_types['Predicted_Type'].value_counts())
