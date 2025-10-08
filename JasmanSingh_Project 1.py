# Project 1
# Jasman Singh, 501180039
# Due October 6, 2025



# Step 1
import pandas as pd

# Reading csv file
data = pd.read_csv("Project 1 Data.csv")



# Step 2
import matplotlib.pyplot as plt
import numpy as np

# Basic information about dataset
print("First 5 rows:\n", data.head())
print("\nDataset info:\n", data.info())
print("\nStep distribution:\n", data['Step'].value_counts().sort_index())
print("\nBasic statistics:\n", data.describe())

# Histograms for each feature 
data[['X', 'Y', 'Z']].hist(bins=25, figsize=(8, 6))
plt.suptitle("Feature Distributions")
plt.show()



# Step 3
import seaborn as sns

# Computing correlation between numeric features
corr = data.corr(numeric_only=True)
print("Correlation matrix:\n", corr, "\n")

# Heatmap for visual correlation understanding
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()



# Step 4
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# Selecting features and target variables
X = data[['X', 'Y', 'Z']]
y = data['Step']

# Splitting data set into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define model parameter grids
param_lr = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
param_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
param_svm = {
    'C': [0.5, 1, 5],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
param_gb = {
    'n_estimators': np.arange(50, 251, 50),
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Run GridSearchCV for 3 models
models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), param_lr),
    "Random Forest": (RandomForestClassifier(random_state=42), param_rf),
    "SVM": (SVC(), param_svm)
}

best_models = {}

# 5-fold cross-validation for training and evaluating each model
for name, (model, params) in models.items():
    grid = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_

    print(f"\n{name} Results:")
    print("Best Parameters:", grid.best_params_)
    print("Best CV Accuracy:", round(grid.best_score_, 4))
    preds = grid.best_estimator_.predict(X_test)
    print("Test Accuracy:", round(accuracy_score(y_test, preds), 4))
    print(classification_report(y_test, preds))

# RandomizedSearchCV for Gradient Boosting
rand_search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_distributions=param_gb,
    n_iter=10,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)
rand_search.fit(X_train, y_train)
best_models["Gradient Boosting"] = rand_search.best_estimator_

print("\nGradient Boosting Results:")
print("Best Params:", rand_search.best_params_)
print("Best CV Accuracy:", round(rand_search.best_score_, 4))
gb_preds = rand_search.best_estimator_.predict(X_test)
print("Test Accuracy:", round(accuracy_score(y_test, gb_preds), 4))
print(classification_report(y_test, gb_preds))



# Step 5
results = []
for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    results.append([
        model_name,
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, average='weighted', zero_division=0),
        recall_score(y_test, y_pred, average='weighted', zero_division=0),
        f1_score(y_test, y_pred, average='weighted', zero_division=0)
    ])

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
print("\nPerformance Summary:\n", results_df)

# Confusion matrices
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()



# Step 6
# Combining random forest and svm while using logistic regression as final estimator
base_learners = [
    ('rf', best_models['Random Forest']),
    ('svm', best_models['SVM'])
]
stacked_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=-1
)
stacked_clf.fit(X_train, y_train)
stack_preds = stacked_clf.predict(X_test)

print("\nStacked Model Results:")
print("Accuracy:", round(accuracy_score(y_test, stack_preds), 4))
print("Precision:", round(precision_score(y_test, stack_preds, average='weighted', zero_division=0), 4))
print("F1 Score:", round(f1_score(y_test, stack_preds, average='weighted', zero_division=0), 4))

# Cofusion matrix for stacked model
cm_stack = confusion_matrix(y_test, stack_preds)
plt.figure(figsize=(7, 5))
sns.heatmap(cm_stack, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix: Stacked Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



# Step 7
import joblib
best_model = best_models['SVM']  # choosing top performer
joblib.dump(best_model, 'maintenance_step_model.joblib')
print("\nBest model saved as 'maintenance_step_model.joblib'")

# Load and predict on new coordinates
loaded = joblib.load('maintenance_step_model.joblib')
test_points = [
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3]
]

preds = loaded.predict(test_points)
print("\nPredicted maintenance steps for given coordinates:")
for i, p in enumerate(preds, start=1):
    print(f"Set {i}: Step {p}")