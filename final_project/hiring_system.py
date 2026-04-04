# ====================================================================================================== #
# Importing Libraries
# ====================================================================================================== #

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# ====================================================================================================== #
# Importing DataSet
# ====================================================================================================== #

df = pd.read_csv(r"datasets/recruitment.csv")

# ====================================================================================================== #
# Creating noise in dataSet
# ====================================================================================================== #

df.loc[1, "Skill_Score"] = np.nan
df.loc[3, "Interview_Score"] = np.nan
df.loc[5, "Years_Experience"] = np.nan

# ====================================================================================================== #
# Info for DataSet
# ====================================================================================================== #

df.info()
print("_"*100)
print(f"Null Count: \n{df.isnull().sum()}")
print("_"*100)

# ====================================================================================================== #
# Cleaning DataSet
# ====================================================================================================== #

cols = ["Skill_Score", "Interview_Score", "Years_Experience"]

for col in cols:
    df[col] = df[col].fillna(df[col].mean())

# ====================================================================================================== #
# Select Input and Output
# ====================================================================================================== #

x = df[["Skill_Score", "Interview_Score", "Years_Experience"]].copy()
y = df["Hired"]

# ====================================================================================================== #
# Train-Test Split
# ====================================================================================================== #

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# ====================================================================================================== #
# Feature Scaling (AFTER SPLIT - NO DATA LEAKAGE)
# ====================================================================================================== #

scaler = MinMaxScaler(feature_range=(0, 100))
scaled_cols = ["Skill_Score", "Interview_Score", "Years_Experience"]

x_train.loc[:, scaled_cols] = scaler.fit_transform(x_train[scaled_cols])
x_test.loc[:, scaled_cols] = scaler.transform(x_test[scaled_cols])

# ====================================================================================================== #
# Logistic Regression Model
# ====================================================================================================== #

log_model = LogisticRegression()
log_model.fit(x_train, y_train)

pred1 = log_model.predict(x_test)

# ====================================================================================================== #
# Decision Tree Model (CONTROLLED)
# ====================================================================================================== #

dt_model = DecisionTreeClassifier(max_depth=3)
dt_model.fit(x_train, y_train)

pred2 = dt_model.predict(x_test)

# ====================================================================================================== #
# Compare Models
# ====================================================================================================== #

print(f"Logistic Regression Model accuracy: {accuracy_score(y_test, pred1)}")
print("_"*100)
print("Logistic Regression Report:")
print(classification_report(y_test, pred1, zero_division=0))
print("_"*100)

print(f"Decision Tree Model accuracy: {accuracy_score(y_test, pred2)}")
print("_"*100)
print("Decision Tree Report:")
print(classification_report(y_test, pred2, zero_division=0))
print("_"*100)

# ====================================================================================================== #
# Save Model and Scaler
# ====================================================================================================== #

joblib.dump(log_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

# ====================================================================================================== #
# Sample Prediction (CLEAN & CONSISTENT)
# ====================================================================================================== #

def predict_hire(skill, interview, experience):
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")

    data = pd.DataFrame(
        [[skill, interview, experience]],
        columns=["Skill_Score", "Interview_Score", "Years_Experience"]
    )

    # Keep feature names -> avoids warnings
    data_scaled = pd.DataFrame(
        scaler.transform(data),
        columns=["Skill_Score", "Interview_Score", "Years_Experience"]
    )

    prediction = model.predict(data_scaled)

    return "Hired" if prediction[0] == 1 else "Not Hired"

# ====================================================================================================== #
# Testing Predictions
# ====================================================================================================== #

print(f"Candidate 1 (85/93/3) : {predict_hire(85, 93, 3)}")
print(f"Candidate 2 (45/38/1) : {predict_hire(45, 38, 1)}")