import pandas as pd
from imblearn.over_sampling import SMOTE
import joblib

db1 = pd.read_csv('data/db1.csv')

X_train = db1.drop("Diabetes_binary", axis= 1)
y_train  = db1['Diabetes_binary']

# Preprocessing imbalance data in dataset 1
smote = SMOTE(random_state=42,sampling_strategy='auto')
X_res, y_res = smote.fit_resample(X_train, y_train)
joblib.dump(y_train.value_counts(normalized = True), 'models/db1_before_smote.pkl')
joblib.dump(y_res.value_counts(normalized = True), 'models/db1_after_smote.pkl')

# Loading y_train before handling imbalance data
# y_train = joblib.load("models/models/db1_before_smote.pkl")
# y_train = joblib.load("models/models/db1_after_smote.pkl")
