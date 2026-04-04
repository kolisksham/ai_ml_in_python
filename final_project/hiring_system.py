# ====================================================================================================== #
# Importing Libraries
# ====================================================================================================== #

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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
# Feature Scaling
# ====================================================================================================== #

scaler = MinMaxScaler(feature_range=(0, 100))
scaled_cols = ["Skill_Score", "Interview_Score"]
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])


# ====================================================================================================== #
# Select Input and Output
# ====================================================================================================== #

x = df[["Skill_Score", "Interview_Score"]]
y = df["Hired"]

# ====================================================================================================== #
# Train-Test Split
# ====================================================================================================== #

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ====================================================================================================== #
# Logistic regression Model
# ====================================================================================================== #

log_model = LogisticRegression()
log_model.fit(x_train, y_train)

pred1 = log_model.predict(x_test)

# ====================================================================================================== #
# Decision Tree Model
# ====================================================================================================== #

dt_model = DecisionTreeClassifier()
dt_model.fit(x_train, y_train)

pred2 = dt_model.predict(x_test)

# ====================================================================================================== #
# Compare Models
# ====================================================================================================== #

print(f"Logistic Regression Model accuracy: {accuracy_score(y_test, pred1)}")
print("_"*100)
print(f"Decision Tree Model accuracy: {accuracy_score(y_test, pred2)}")
print("_"*100)


# ====================================================================================================== #
# Sample Prediction
# ====================================================================================================== #

def predict_hire(skill, interview):
    data = pd.DataFrame([[skill, interview]], columns = ["Skill_Score", "Interview_Score"])
    data_scaled = scaler.transform(data[["Skill_Score", "Interview_Score"]])
    data_scaled = pd.DataFrame(data_scaled, columns = ["Skill_Score", "Interview_Score"])
    prediction = log_model.predict(data_scaled)

    if prediction[0] == 1:
        return("Hired")
    else:
        return("Not Hired")

print(f"Candidate 1 (85/93) : {predict_hire(85, 93)}")
print(f"Candidate 2 (45/38) : {predict_hire(45, 38)}")
