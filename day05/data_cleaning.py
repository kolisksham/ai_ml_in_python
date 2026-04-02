import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"datasets/recruitment.csv")

# ====================================================================================================== #
# Basic Info about Data and check Null Count
# ====================================================================================================== #

df.info()
print('_'*100)

print(f"Null Value Count: \n{df.isnull().sum()}")
print('_'*100)

# ====================================================================================================== #
# Deliberately Create Missing Values
# ====================================================================================================== #

df.loc[1, "Skill_Score"] = np.nan
df.loc[3, "Interview_Score"] = np.nan
df.loc[5, "Years_Experience"] = np.nan

df.info()
print('_'*100)

print(f"Null Value Count: \n{df.isnull().sum()}")
print('_'*100)

# ====================================================================================================== #
# Fill Missing Values with Mean
# ====================================================================================================== #

df["Skill_Score"] = df["Skill_Score"].fillna(df["Skill_Score"].mean())
df["Interview_Score"] = df["Interview_Score"].fillna(df["Interview_Score"].mean())
df["Years_Experience"] = df["Years_Experience"].fillna(df["Years_Experience"].mean())

df = df.dropna(axis=1, how="all")

# ====================================================================================================== #
# Logistic Regression Model
# ====================================================================================================== #

x = df[["Skill_Score", "Interview_Score"]]
y = df["Hired"]

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)

pred = model.predict(x_test)

print(f"Logistic Regression Model Accuracy : {accuracy_score(y_test, pred)}")
print('_'*100)



