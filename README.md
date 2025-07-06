# PREDICTIVE-ANALYSIS-USING-MACHINE-LEARNING

COMPANY NAME - CODTECH IT SOLUTIONS

NAME - Manthan Gupta

INTERN ID - CT06DG1412

DOMAIN NAME - DATA ANALYSIS

DURATION - 6 WEEKS(June 14th 2025 to July 29th 2025)

MENTOR - NEELA SANTHOSH KUMAR

Description:

This notebook presents a complete machine learning pipeline designed to predict future outcomes based on historical data. It combines data preprocessing, exploration, modeling, evaluation, and interpretation. Here’s a detailed breakdown of what the code likely includes:

🔹 1. Library Imports
At the start, the notebook imports necessary libraries such as:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

🔹 2. Data Loading
df = pd.read_csv("filename.csv")
The dataset is imported—typically in CSV format—for analysis.
This section may also include:
df.head(): Preview first few rows.
df.shape: Dimensions of the data.
df.info(): Types and null values.

🔹 3. Exploratory Data Analysis (EDA)
Purpose: To understand patterns, distributions, and relationships in the data.
Typical steps:
Histograms for numerical columns
Count plots for categorical features
Correlation heatmap (sns.heatmap(df.corr()))
Box plots to detect outliers
This phase is crucial for identifying:
Skewed distributions
Feature importance
Redundant or missing features

🔹 4. Data Preprocessing
This includes:
🧼 Cleaning:
Dropping irrelevant or missing data: df.dropna() or df.fillna()
🏷️ Encoding:
Label Encoding for categorical variables:
le = LabelEncoder()
df['column'] = le.fit_transform(df['column'])
📏 Feature Scaling:
StandardScaler() or MinMaxScaler() for normalization

🔹 5. Splitting Dataset
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Target variable is separated from features.
Data is split into training and testing sets.

🔹 6. Model Building
A machine learning model (like RandomForestClassifier, LogisticRegression, or SVC) is trained:
model = RandomForestClassifier()
model.fit(X_train, y_train)

🔹 7. Prediction and Evaluation
Model is evaluated using:
y_pred = model.predict(X_test)
Then metrics are calculated:
Accuracy Score
Classification Report: Precision, Recall, F1-score
Confusion Matrix
Possibly ROC-AUC or Precision-Recall Curve
Visualizations:
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)

🔹 8. Feature Importance (for tree models)
importances = model.feature_importances_
Often plotted as a bar chart to show which features contribute most to predictions.

🔹 9. Optional: Hyperparameter Tuning
If included, tuning may use:
GridSearchCV
RandomizedSearchCV
Purpose: Improve model performance by finding the best parameter combinations.

🔹 10. Conclusion
Final insights or interpretations from the analysis:
Model accuracy
Most important predictors
Suggestions for deployment or further improvement


Output:

<img width="1516" height="304" alt="Image" src="https://github.com/user-attachments/assets/48280b71-cc7d-4068-bcf4-4b7843940a12" />

<img width="719" height="282" alt="Image" src="https://github.com/user-attachments/assets/79342875-23e1-4af9-b1bb-512aec4d030a" />

<img width="724" height="462" alt="Image" src="https://github.com/user-attachments/assets/ba92b2d9-58a2-4d34-a6d6-94ff97c76e72" />

<img width="719" height="470" alt="Image" src="https://github.com/user-attachments/assets/31a5584d-b4f3-4e6b-83a2-c2d1551aa38f" />

