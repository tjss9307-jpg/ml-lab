Part 1: Standalone Data Preprocessing (Run this First)
Run this cell to produce clean, scaled feature matrices X_train_scaled and X_test_scaled without pipelines.

Python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load data
df = pd.read_csv('your_dataset.csv')
target_col = 'target'

X = df.drop(columns=[target_col])
y = df[target_col]

# 2. Split FIRST to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Handle Missing Values (Separately on Train & Test)
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns

# Numerical: fill using training set median
for col in num_cols:
    median_val = X_train[col].median()
    X_train[col] = X_train[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

# Categorical: fill with 'missing' category
for col in cat_cols:
    X_train[col] = X_train[col].fillna('missing')
    X_test[col] = X_test[col].fillna('missing')

# 4. Dummy Encoding with pd.get_dummies & align columns
X_train_encoded = pd.get_dummies(X_train, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, drop_first=True)

# Ensure train and test share identical columns
X_train_encoded, X_test_encoded = X_train_encoded.align(
    X_test_encoded, join='left', axis=1, fill_value=0
)

# 5. Feature Scaling (Fit on Train ONLY, transform both)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

print(f"Preprocessed Train shape: {X_train_scaled.shape}")
print(f"Preprocessed Test shape:  {X_test_scaled.shape}")
Part 2: Logistic Regression (with GridSearchCV)
Parameters tuned:

C: Inverse of regularization strength. Smaller values mean stronger regularization (preventing overfitting).

penalty: Regularization norm ('l1' sets unimportant weights to 0, 'l2' shrinks weights smoothly).

solver: Optimization algorithm ('saga' and 'liblinear' support both l1 and l2).

Python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# 1. Initialize base model (no pipeline)
lr_model = LogisticRegression(max_iter=1000, random_state=42)

# 2. Define the hyperparameter grid
# Notice: No 'model__' prefix needed because there is no pipeline!
param_grid_lr = {
    'C': [0.01, 0.1, 1.0, 10.0],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# 3. Setup GridSearchCV
lr_grid = GridSearchCV(
    estimator=lr_model,
    param_grid=param_grid_lr,
    cv=5,                 # 5-fold cross-validation
    scoring='accuracy',   # Can change to 'f1', 'roc_auc', etc.
    n_jobs=-1,
    verbose=1
)

# 4. Fit on preprocessed training data
print("Tuning Logistic Regression...")
lr_grid.fit(X_train_scaled, y_train)

# 5. Inspect best parameters and CV score
print("\n--- Logistic Regression Best Results ---")
print(f"Best Hyperparameters: {lr_grid.best_params_}")
print(f"Best 5-Fold CV Score: {lr_grid.best_score_:.4f}")

# 6. Evaluate automatically refitted model on the test set
best_lr = lr_grid.best_estimator_
y_pred_lr = best_lr.predict(X_test_scaled)

print(f"Test Set Accuracy:   {accuracy_score(y_test, y_pred_lr):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))
Part 3: Support Vector Machine / SVM (with GridSearchCV)
Parameters tuned:

C: Regularization parameter. High values try to classify every point correctly (risk overfitting); low values allow margin violations for a smoother boundary.

kernel: Determines how the data is mapped to higher dimensions ('linear', 'rbf', or 'poly').

gamma: Kernel coefficient for 'rbf' and 'poly'. High values mean a tighter decision boundary around individual points; low values mean a broader, smoother boundary.

Python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# 1. Initialize base SVM model (probability=True allows ROC-AUC & predict_proba)
svm_model = SVC(probability=True, random_state=42)

# 2. Define the hyperparameter grid
param_grid_svm = {
    'C': [0.1, 1.0, 10.0],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto', 0.01, 0.1]
}

# 3. Setup GridSearchCV
svm_grid = GridSearchCV(
    estimator=svm_model,
    param_grid=param_grid_svm,
    cv=5,                 # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# 4. Fit on preprocessed training data
print("Tuning SVM...")
svm_grid.fit(X_train_scaled, y_train)

# 5. Inspect best parameters and CV score
print("\n--- SVM Best Results ---")
print(f"Best Hyperparameters: {svm_grid.best_params_}")
print(f"Best 5-Fold CV Score: {svm_grid.best_score_:.4f}")

# 6. Evaluate automatically refitted model on the test set
best_svm = svm_grid.best_estimator_
y_pred_svm = best_svm.predict(X_test_scaled)

print(f"Test Set Accuracy:   {accuracy_score(y_test, y_pred_svm):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_





















1. K-Nearest Neighbors (KNN)In KNeighborsClassifier, the model classifies a point based on the labels of its nearest neighbors in feature space.Key Hyperparameters to Tune:n_neighbors:What it does: The number of neighbors ($K$) to look at (e.g., [3, 5, 7, 11, 15]).Impact: Small $K$ (e.g., 1 or 3) leads to complex decision boundaries that risk overfitting (sensitive to noise). Large $K$ leads to smoother boundaries that risk underfitting. Always try odd numbers for binary classification to break ties.weights:What it does: 'uniform' vs. 'distance'.Impact: 'uniform' weights all neighbors equally. 'distance' gives closer neighbors more influence on the final vote than distant ones.metric (and p):What it does: The distance function used to measure proximity ('euclidean', 'manhattan', or 'minkowski').Impact: If metric='minkowski', setting p=1 corresponds to Manhattan distance (sum of absolute differences), and p=2 corresponds to Euclidean distance (straight-line distance).KNN Code (Standalone, No Pipeline):Pythonfrom sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# 1. Define base model
knn_model = KNeighborsClassifier()

# 2. Hyperparameter Grid
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# 3. Grid Search
knn_grid = GridSearchCV(
    estimator=knn_model,
    param_grid=param_grid_knn,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

knn_grid.fit(X_train_scaled, y_train)

print(f"Best KNN Params: {knn_grid.best_params_}")
print(f"Best CV Score:  {knn_grid.best_score_:.4f}")
Important Note for KNN: Always make sure your features are scaled (StandardScaler or MinMaxScaler) before training KNN, because distance calculations are completely distorted by unscaled numeric ranges.2. Naive Bayes (NB)Naive Bayes algorithms are probabilistic classifiers based on Bayes' theorem with the "naive" assumption of feature independence. Because NB computes closed-form probability distributions directly from data, it has very few hyperparameters to tune.Which version of Naive Bayes you use depends on the nature of your features:A. Gaussian Naive Bayes (GaussianNB)Used for continuous numerical features (assumes features follow a normal Gaussian distribution).Key Hyperparameter:var_smoothing: Portion of the largest variance of all features that is added to variances for calculation stability (default is 1e-9).Why tune it: If standard distributions don't fit smoothly, searching logarithmic values (e.g., np.logspace(-11, -5, 7)) widens or smooths the curve and prevents zero-probability issues.Pythonfrom sklearn.naive_bayes import GaussianNB
import numpy as np

gnb_model = GaussianNB()

param_grid_gnb = {
    'var_smoothing': np.logspace(-11, -5, 7) # e.g. [1e-11, 1e-10, ..., 1e-5]
}

gnb_grid = GridSearchCV(estimator=gnb_model, param_grid=param_grid_gnb, cv=5, scoring='accuracy')
gnb_grid.fit(X_train_scaled, y_train)

print(f"Best GaussianNB Params: {gnb_grid.best_params_}")
B. Multinomial Naive Bayes (MultinomialNB) & Bernoulli Naive Bayes (BernoulliNB)MultinomialNB is used for discrete counts/frequencies (e.g., word counts; requires non-negative values).BernoulliNB is used for binary 0/1 boolean features (like one-hot dummy columns).Key Hyperparameters:alpha: Laplace/Lidstone smoothing parameter (0.01, 0.1, 0.5, 1.0, 2.0).Why tune it: Prevents zero-probability errors when a category or word was never seen in a given class during training. alpha=1.0 is standard Laplace smoothing.fit_prior: [True, False]Why tune it: True learns prior probabilities $P(Y)$ directly from class proportions in the data. False assumes uniform priors (treats all classes as equally likely regardless of class balance).Pythonfrom sklearn.naive_bayes import BernoulliNB

bnb_model = BernoulliNB()

param_grid_bnb = {
    'alpha': [0.01, 0.1, 0.5, 1.0, 2.0],
    'fit_prior': [True, False]
}

bnb_grid = GridSearchCV(estimator=bnb_model, param_grid=param_grid_bnb, cv=5, scoring='accuracy')
bnb_grid.fit(X_train_scaled, y_train)

print(f"Best BernoulliNB Params: {bnb_grid.best_params_}")




















from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, roc_auc_score, cohen_kappa_score, 
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
import matplotlib.pyplot as plt

# 1. Extract the best trained model from GridSearchCV
best_model = grid_search.best_estimator_
print("Best Model extracted successfully!")

# 2. Get Predictions and Probabilities on the Test Set
y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)[:, 1] # Probability for positive class (1)

# 3. Compute All Evaluation Metrics
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cross_entropy = log_loss(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)
kappa = cohen_kappa_score(y_test, y_pred)

# 4. Print Summary Table
print("=" * 45)
print("          MODEL EVALUATION METRICS           ")
print("=" * 45)
print(f"Accuracy:             {acc:.4f}")
print(f"Precision:            {precision:.4f}")
print(f"Recall:               {recall:.4f}")
print(f"F1 Score:             {f1:.4f}")
print(f"Cross-Entropy Loss:   {cross_entropy:.4f}")
print(f"ROC AUC:              {roc_auc:.4f}")
print(f"Cohen's Kappa:        {kappa:.4f}")
print("=" * 45)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 5. Plot Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
disp.plot(cmap='Blues', colorbar=False, ax=plt.gca())
plt.title("Confusion Matrix", fontsize=14)
plt.grid(False)
plt.show()