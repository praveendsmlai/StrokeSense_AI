NeuroGuard AI: Stroke Risk Prediction System

Project Overview

NeuroGuard AI is a machine learning-based system designed to predict the risk of stroke in patients using clinical and demographic data. Early prediction of stroke can help in timely medical intervention and potentially save lives.

This project focuses on handling imbalanced healthcare data, applying data preprocessing pipelines, and comparing multiple machine learning models to identify the best-performing approach.


Dataset Information
Dataset Name: Stroke Prediction Dataset
File Used: train_2v.csv
Records: ~43,000+
Target Variable: stroke (0 = No, 1 = Yes)


Features:
Feature	Description
id	Unique identifier
gender	Male / Female
age	Age of patient
hypertension	0 = No, 1 = Yes
heart_disease	0 = No, 1 = Yes
ever_married	Yes / No
work_type	Job type
Residence_type	Urban / Rural
avg_glucose_level	Blood glucose level
bmi	Body Mass Index
smoking_status	Smoking habits
stroke	Target variable

Tech Stack
Language: Python
Libraries:
pandas, numpy
scikit-learn
imbalanced-learn

🔄 Workflow
1. Data Preprocessing
Missing values handled using:
Median (numerical)
Most frequent (categorical)
Feature scaling with StandardScaler
Encoding using OneHotEncoder
Pipeline built using ColumnTransformer

3. Train-Test Split
75% Training / 25% Testing
Stratified sampling ensures balanced distribution

5. Handling Imbalance
Applied SMOTE (Synthetic Minority Oversampling Technique)
Used only on training data to prevent data leakage

7. Models Implemented
Logistic Regression (class_weight="balanced")
Naive Bayes
K-Nearest Neighbors
Decision Tree (balanced)
Random Forest (balanced)

9. Threshold Tuning
Custom threshold: 0.3
Improves detection of stroke cases (important in healthcare)

📊 Evaluation Metrics
Accuracy
F1 Score ⭐ (Primary metric)
ROC-AUC Score
Classification Report
Confusion Matrix

🏆 Model Selection
Best model selected based on highest F1-score
Helps balance precision and recall for imbalanced data

📈 Key Insights
Dataset is highly imbalanced (very few stroke cases)
SMOTE significantly improves minority class detection
Lower threshold increases recall (reduces false negatives)
Tree-based models and Logistic Regression perform effectively
