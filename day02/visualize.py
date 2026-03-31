import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("datasets/recruitment.csv")

print(f"Hired:\n{df["Hired"].value_counts()}")
print("_"*100)

colors = df["Hired"]

plt.scatter(df["Skill_Score"], df["Interview_Score"], c=colors)
plt.xlabel("Skills Score")
plt.ylabel("Interview Score")
plt.title("Hiring Selection Graph")
plt.show()

print(f"Averages:\n{df.groupby('Hired')[['Skill_Score', 'Interview_Score']].mean()}")

df["Total_Score"] = df[["Skill_Score", "Interview_Score"]].mean(axis=1)
plt.hist(df["Total_Score"])
plt.title("Total Score Distribution")
plt.show()

plt.scatter(df["Total_Score"], df["Hired"], c=colors)
plt.title("Total Score VS Hiring Graph")
plt.show()