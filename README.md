# 🚀 AI-ML Playground

This repository documents my hands-on journey of learning Machine Learning by building and experimenting daily.

---

## 🎯 Goal

To become job-ready in AI/ML by:

* Building real-world projects
* Understanding core concepts deeply
* Practicing consistently through structured daily challenges

---

## 📅 Progress Log

### ✅ Day 1: Basic ML Model

* Built a hiring prediction model using Logistic Regression
* Learned:

  * Features vs Target
  * Train-Test Split
  * Model training and prediction
* Successfully implemented a basic ML pipeline

---

### ✅ Day 2: Data Visualization & Insights

* Visualized relationships between features
* Created scatter plots:

  * Skill Score vs Interview Score
* Used color mapping to distinguish hiring outcomes
* Analyzed distributions using histograms
* Observed clear patterns in hiring decisions

---

### ✅ Day 3: Feature Engineering & Scaling

* Created engineered features:

  * `Total_Score` (combined performance metric)
  * `Experience_Score` (scaled using MinMaxScaler)
* Applied feature scaling for consistent comparison
* Compared multiple models:

  * Skill + Interview → **1.0 accuracy**
  * Experience only → **0.5 accuracy**
  * Total Score → **0.75 accuracy**
  * All features combined → **0.75 accuracy**
* Key Learnings:

  * More features ≠ better performance
  * Weak features can degrade model accuracy
  * Feature scaling improves fairness across features
  * Feature compression can lead to information loss

---

### ✅ Day 4: Model Comparison (Logistic vs Decision Tree)

* Implemented Decision Tree classifier
* Compared with Logistic Regression
* Observed differences in model behavior:

  * Logistic Regression → linear decision boundary
  * Decision Tree → rule-based decisions
* Visualized Decision Tree structure
* Experimented with tree depth (`max_depth`)
* Key Learnings:

  * Different models think differently
  * Decision Trees can handle non-linear patterns
  * Risk of overfitting with deeper trees
  * Model selection depends on data characteristics

---

### ✅ Day 5: Data Cleaning & Real-World Simulation

* Introduced missing values manually to simulate real-world data
* Handled missing data using mean imputation
* Built a simple data cleaning pipeline
* Key Learnings:

  * Real-world data is messy
  * Data cleaning is as important as modeling
  * Models fail without proper preprocessing

---

### ✅ Day 6: End-to-End ML Project (Hiring Prediction System)

* Built a complete ML system:

  * Data cleaning → preprocessing → training → prediction
* Fixed **data leakage issue** (scaling after train-test split)
* Compared models with proper evaluation:

  * Accuracy + Precision + Recall + F1-score
* Controlled overfitting in Decision Tree (`max_depth`)
* Implemented model persistence using `joblib`
* Created reusable prediction function
* Key Learnings:

  * Importance of proper ML workflow
  * Data leakage can give false results
  * Evaluation metrics matter beyond accuracy
  * ML systems should be reusable and consistent

---

### ✅ Day 7,8,9: Theory Revision/Break

* Taking a Break after 6 days of fast-paced Learning.

---

## 🧠 Key Concepts Learned

* Machine Learning pipeline (data → model → prediction)
* Feature engineering and its impact
* Feature scaling using MinMaxScaler
* Model evaluation using accuracy, precision, recall, F1-score
* Handling missing data
* Data leakage and how to prevent it
* Model comparison and behavior analysis
* Overfitting and model control
* Model persistence (saving/loading models)

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
├── day03_improvement/     # Feature engineering & scaling
├── day04_models/          # Model comparison (Logistic vs Tree)
├── day05_cleaning/        # Data cleaning & preprocessing
├── final_project/         # End-to-end ML system (Hiring Prediction)
├── models/                # Saved models (.pkl files)
├── utils/                 # Helper functions (future use)
├── notebooks/             # Experiments & notes
└── README.md
```

---

## 🔥 Learning Approach

* Learn by building (not just watching)
* Experiment and analyze results
* Focus on daily consistency
* Understand *why* models behave a certain way

---

## ⚡ Key Insight So Far

> The best model is not the most complex —
> it’s the one with the right features and understanding.

---

## 🚀 Next Steps

* Build a Flask-based ML API
* Deploy the model as a usable application
* Improve preprocessing using pipelines
* Add more realistic datasets
* Prepare project for portfolio & interviews

---

## 🧠 Motto

Consistency > Motivation

---

## 📌 Status

🟢 In Progress — Day 6 Complete
🔥 Next: Day 7 — Model Deployment (Flask API)
