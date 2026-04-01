# 🚀 AI-ML Playground

This repository documents my hands-on journey of learning Machine Learning by building and experimenting daily.

---

## 🎯 Goal

To become job-ready in AI/ML by:

* Building real projects
* Understanding core concepts deeply
* Practicing consistently with structured daily tasks

---

## 📅 Progress Log

### ✅ Day 1: Basic ML Model

* Built a hiring prediction model using Logistic Regression
* Learned:

  * Features vs Target
  * Train-Test Split
  * Model training and prediction
* Achieved first working ML pipeline

---

### ✅ Day 2: Data Visualization & Insights

* Visualized relationships between features
* Created scatter plots for:

  * Skill Score vs Interview Score
* Used color mapping to distinguish hiring outcomes
* Analyzed feature distributions using histograms
* Observed patterns in hiring decisions

---

### ✅ Day 3: Feature Engineering & Model Improvement

* Created new features:

  * `Total_Score` (combined performance metric)
  * `Experience_Score` (scaled using MinMaxScaler)
* Applied feature scaling for consistency
* Compared multiple models:

  * Skill + Interview → Accuracy: **1.0**
  * Experience only → Accuracy: **0.5**
  * Total Score → Accuracy: **0.75**
  * All features combined → Accuracy: **0.75**
* Key Learnings:

  * More features ≠ better performance
  * Weak features can degrade model accuracy
  * Feature selection is critical in ML
  * Information loss occurs when compressing features

---

## 🧠 Key Concepts Learned

* Machine Learning pipeline (data → model → prediction)
* Feature engineering and its impact
* Feature scaling using MinMaxScaler
* Model evaluation using accuracy
* Feature importance and noise vs signal

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

---

## 📁 Project Structure

```
AI-ML-Playground/
│
├── datasets/              # Raw datasets
├── day01_basics/          # Basic ML model
├── day02_visualization/   # Data visualization & insights
├── day03_improvement/     # Feature engineering & experiments
├── models/                # Saved models (future use)
├── utils/                 # Helper functions (future use)
├── notebooks/             # Experiments & notes
└── README.md
```

---

## 🔥 Learning Approach

* Learn by doing (not just watching)
* Experiment and analyze results
* Focus on small daily improvements
* Build intuition behind models

---

## ⚡ Key Insight So Far

> The best model is not the most complex —
> it’s the one with the right features.

---

## 🚀 Next Steps

* Compare different ML models (Decision Tree vs Logistic Regression)
* Understand model behavior and decision boundaries
* Build a complete end-to-end ML project

---

## 🧠 Motto

Consistency > Motivation

---

## 📌 Status

🟢 In Progress — Day 3 Complete
🔜 Moving to Day 4: Model Comparison
