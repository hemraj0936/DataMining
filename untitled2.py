#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 12:18:38 2024

@author: hemrajsaini
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load the dataset (adjust this line according to your file path and format)
df = pd.read_excel("great_data.xlsx")

# Drop 'userId' and 'state' columns as instructed
df_processed = df.drop(columns=['userId', 'state'])

# Drop rows with any missing values
df_processed = df_processed.dropna()

# Convert categorical variables into dummy/indicator variables
df_processed = pd.get_dummies(df_processed, drop_first=True)

# Separate the data into features and target variable
y = df_processed['churn']
X = df_processed.drop(columns='churn')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Initialize the DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(random_state=1)

# Fit the model
dt_classifier.fit(X_train, y_train)

# Predictions
y_pred_train = dt_classifier.predict(X_train)
y_pred_test = dt_classifier.predict(X_test)

# Evaluation using F1 score
f1_score_train = f1_score(y_train, y_pred_train, average='macro')
f1_score_test = f1_score(y_test, y_pred_test, average='macro')

print("Initial model F1 scores - Train: ", f1_score_train, ", Test: ", f1_score_test)

# Hyperparameter tuning
parameter_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8],  # Example range, adjust as needed
    'min_samples_split': [5, 10, 15, 20]  # Example range, adjust as needed
}

# Setting up GridSearchCV
grid_search = GridSearchCV(dt_classifier, parameter_grid, verbose=3, scoring='f1_macro', cv=5)  # Using 5-fold CV

# Fitting GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters
best_params = grid_search.best_params_
print("Best parameters: ", best_params)

# Building decision tree model with best parameters
dt_classifier_tuned = DecisionTreeClassifier(**best_params, random_state=1)
dt_classifier_tuned.fit(X_train, y_train)

# Evaluate on training data with tuned model
y_pred_train_tuned = dt_classifier_tuned.predict(X_train)
f1_score_train_tuned = f1_score(y_train, y_pred_train_tuned, average='macro')

# Evaluate on testing data with tuned model
y_pred_test_tuned = dt_classifier_tuned.predict(X_test)
f1_score_test_tuned = f1_score(y_test, y_pred_test_tuned, average='macro')

print("Tuned model F1 scores - Train: ", f1_score_train_tuned, ", Test: ", f1_score_test_tuned)

# Optional: Plot the decision tree of the tuned model
plt.figure(figsize=(20,10))
plot_tree(dt_classifier_tuned, filled=True, feature_names=X.columns)
plt.show()
