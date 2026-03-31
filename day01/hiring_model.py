import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df = pd.read_csv("datasets/recruitment.csv")


print(f"Data:\n{df.head()}")
print("_"*100)

x = df[["Skill_Score", "Interview_Score"]]
y = df["Hired"]

x_train, x_test, y_train, y_test = train_test_split(x, y)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")
print("_"*100)
print(df.groupby("Hired")["Skill_Score"].mean())
print("_"*100)