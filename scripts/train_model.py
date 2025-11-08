import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import joblib
import os

# Paths
DATA_PATH = '/Users/Jee/College/II Year/ Odd Sem(III)/Technical/Data Science/flight_delay_project/data/flights.csv'
MODEL_PATH = '/Users/Jee/College/II Year/ Odd Sem(III)/Technical/Data Science/flight_delay_project/models/flight_delay_model.pkl'
SCALER_PATH = '/Users/Jee/College/II Year/ Odd Sem(III)/Technical/Data Science/flight_delay_project/models/scaler.pkl'

# Load dataset
df = pd.read_csv(DATA_PATH)
df['DELAYED'] = np.where(df['ARR_DELAY'] > 15, 1, 0)

# Features
features = ['CRS_DEP_TIME','DISTANCE','TAXI_OUT','TAXI_IN']
X = df[features]
y = df['DELAYED']

# Handle missing values
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# XGBoost model
xgb_model = xgb.XGBClassifier(tree_method='hist', eval_metric='logloss', n_jobs=-1)

param_grid = {
    'max_depth': [6,8],
    'learning_rate': [0.05,0.1],
    'n_estimators': [200,400],
    'subsample': [0.7,0.8],
    'colsample_bytree': [0.7,0.8]
}

grid = GridSearchCV(xgb_model, param_grid, scoring='recall', cv=3, verbose=1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Save model and scaler
os.makedirs('/Users/Jee/College/II Year/ Odd Sem(III)/Technical/Data Science/flight_delay_project/models', exist_ok=True)
joblib.dump(best_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

# Metrics
y_pred = best_model.predict(X_test)
print("✅ Model trained!")
print(f"Best hyperparameters: {grid.best_params_}")
print(f"Accuracy: {accuracy_score(y_test,y_pred):.4f}")
print(classification_report(y_test,y_pred))