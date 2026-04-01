import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"datasets/recruitment.csv")

scaler = MinMaxScaler(feature_range=( 0, 100))
df["Experience_Score"] = scaler.fit_transform(df[["Years_Experience"]])

df["Total_Score"] = (
    0.45 * df["Skill_Score"] +
    0.45 * df["Interview_Score"] +
    0.10 * df["Experience_Score"]
)

# Target
y = df["Hired"]

# ====================================================================================================== #
# MODEL 1: Skill + Interview
# ====================================================================================================== #
x1 = df[["Skill_Score", "Interview_Score"]]

x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y, test_size = 0.2, random_state=42)

model1 = LogisticRegression()
model1.fit(x_train1, y_train1)

pred1 = model1.predict(x_test1)
print(f"Model 1 (Skills + Interview): {accuracy_score(y_test1, pred1)}")

# ====================================================================================================== #
# MODEL 2: Experience Score
# ====================================================================================================== #

x2 = df[["Experience_Score"]]

x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y, test_size=0.2, random_state=42)

model2 = LogisticRegression()
model2.fit(x_train2, y_train2)

pred2 = model2.predict(x_test2)
print(f"Model 2 (Experience): {accuracy_score(y_test2, pred2)}")

# ====================================================================================================== #
# MODEL 3: Total Score
# ====================================================================================================== #

x3 = df[["Total_Score"]]

x_train3, x_test3, y_train3, y_test3 = train_test_split(x3, y, test_size=0.2, random_state=42)

model3 = LogisticRegression()
model3.fit(x_train3, y_train3)

pred3 = model3.predict(x_test3)
print(f"Model 3 (Total Score) : {accuracy_score(y_test3, pred3)}")

# ====================================================================================================== #
# MODEL 4: Skill + Interview + Experience
# ====================================================================================================== #

X4 = df[["Skill_Score", "Interview_Score", "Experience_Score"]]

x_train4, x_test4, y_train4, y_test4 = train_test_split(X4, y, test_size=0.2, random_state=42)

model4 = LogisticRegression()
model4.fit(x_train4, y_train4)

pred4 = model4.predict(x_test4)

print("Model 4 (All Features):", accuracy_score(y_test4, pred4))