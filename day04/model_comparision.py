import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"datasets/recruitment.csv")

# input
x = df[["Skill_Score", "Interview_Score"]]

# output
y = df["Hired"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ====================================================================================================== #
# MODEL 1: Logostic Regression Model
# ====================================================================================================== #

logi_model = LogisticRegression()
logi_model.fit(x_train, y_train)

pred1 = logi_model.predict(x_test)

print(f"Logistic Regression Model Accuracy: {accuracy_score(y_test, pred1)}")
print("_"*100)

# ====================================================================================================== #
# MODEL 2: Decision Tree Model
# ====================================================================================================== #

dt_model = DecisionTreeClassifier(max_depth=2)
dt_model.fit(x_train, y_train)

pred2 = dt_model.predict(x_test)

print(f"Decision Tree Model Accuracy: {accuracy_score(y_test, pred2)}")
print("_"*100)

# ====================================================================================================== #
# Decision Tree Model Visualization
# ====================================================================================================== #

plt.figure(figsize=(10, 6))
plot_tree(dt_model, feature_names=x.columns, filled=True)
plt.show()