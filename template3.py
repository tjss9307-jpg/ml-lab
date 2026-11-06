import pandas as pd
import numpy as np

# Scikit-Learn Imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, precision_recall_curve
)

# =====================================================================
# CELL 1: LOAD DATA
# =====================================================================
print("--- 1. LOADING DATA ---")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')  # Professor's unseen file with NO labels

target_col = 'target' # <-- CHANGE THIS to your actual target column name

X_full = train_df.drop(columns=[target_col])
y_full = train_df[target_col]
X_test_exam = test_df.copy()

print(f"Training Features Shape: {X_full.shape}")
print(f"Exam Test Shape:       {X_test_exam.shape}")


# =====================================================================
# CELL 2: INTERNAL TRAIN / VALIDATION SPLIT
# =====================================================================
print("\n--- 2. CREATING INTERNAL VALIDATION SPLIT ---")
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)


# =====================================================================
# CELL 3: DEFINE PREPROCESSING PIPELINE (ColumnTransformer)
# =====================================================================
print("\n--- 3. SETTING UP PREPROCESSING PIPELINE ---")

# A. Numeric Pipeline: Fill missing with mean -> Standard scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# B. Categorical Pipeline: Fill missing with 'missing' -> One-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# C. ColumnTransformer to route columns automatically by data type
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
        ('cat', categorical_transformer, make_column_selector(dtype_include=['object', 'category']))
    ],
    remainder='drop'
)


# =====================================================================
# CELL 4: REUSABLE EVALUATION FUNCTION (Prints All Metrics cleanly)
# =====================================================================
def evaluate_and_print_model(y_true, y_pred, y_proba, model_name="Model"):
    """Calculates and neatly prints all standard classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, y_proba) if y_proba is not None else 0.0
    
    print(f"\n==========================================")
    print(f"          {model_name} METRICS           ")
    print(f"==========================================")
    print(f"Accuracy:      {acc:.4f}")
    print(f"Precision:     {prec:.4f}")
    print(f"Recall:        {rec:.4f}")
    print(f"F1-Score:      {f1:.4f}")
    print(f"ROC-AUC:       {roc:.4f}")
    print("------------------------------------------")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("==========================================")
    
    return {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1, 'ROC-AUC': roc}


# =====================================================================
# CELL 5: MODEL 1 - LOGISTIC REGRESSION (Pipeline + GridSearchCV)
# =====================================================================
print("\n--- 5. TRAINING LOGISTIC REGRESSION ---")

lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

lr_param_grid = {'classifier__C': [0.01, 0.1, 1.0, 10.0]}

lr_grid = GridSearchCV(estimator=lr_pipeline, param_grid=lr_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
lr_grid.fit(X_train, y_train)

best_lr = lr_grid.best_estimator_
print(f"Best LR Params: {lr_grid.best_params_} | Best CV Accuracy: {lr_grid.best_score_:.4f}")


# =====================================================================
# CELL 6: MODEL 2 - SUPPORT VECTOR MACHINE / SVM (Pipeline + GridSearchCV)
# =====================================================================
print("\n--- 6. TRAINING SVM ---")

svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(probability=True, random_state=42)) # probability=True is needed for predict_proba
])

svm_param_grid = {
    'classifier__C': [0.1, 1.0, 10.0],
    'classifier__kernel': ['linear', 'rbf']
}

svm_grid = GridSearchCV(estimator=svm_pipeline, param_grid=svm_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
svm_grid.fit(X_train, y_train)

best_svm = svm_grid.best_estimator_
print(f"Best SVM Params: {svm_grid.best_params_} | Best CV Accuracy: {svm_grid.best_score_:.4f}")


# =====================================================================
# CELL 7: COMPARE MODELS ON VALIDATION SET
# =====================================================================
print("\n--- 7. COMPARING MODELS ON VALIDATION SET ---")

lr_val_preds = best_lr.predict(X_val)
lr_val_proba = best_lr.predict_proba(X_val)[:, 1]

svm_val_preds = best_svm.predict(X_val)
svm_val_proba = best_svm.predict_proba(X_val)[:, 1]

# Evaluate both using our custom function
lr_metrics = evaluate_and_print_model(y_val, lr_val_preds, lr_val_proba, "Logistic Regression")
svm_metrics = evaluate_and_print_model(y_val, svm_val_preds, svm_val_proba, "Support Vector Machine (SVM)")

# Summary Comparison Table
comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Logistic Regression': list(lr_metrics.values()),
    'SVM': list(svm_metrics.values())
})

print("\n--- SUMMARY COMPARISON TABLE ---")
print(comparison_df.to_string(index=False))

# Automatically select the winning model based on Validation F1-score
if lr_metrics['F1-Score'] >= svm_metrics['F1-Score']:
    winning_model = best_lr
    winning_name = "Logistic Regression"
    winning_val_proba = lr_val_proba
else:
    winning_model = best_svm
    winning_name = "SVM"
    winning_val_proba = svm_val_proba

print(f"\n>>> WINNING MODEL CHOSEN FOR SUBMISSION: {winning_name}")


# =====================================================================
# CELL 8: THRESHOLD TUNING (Precision-Recall Curve for Imbalanced Data)
# =====================================================================
print("\n--- 8. TUNING DECISION THRESHOLD ---")

precisions, recalls, thresholds = precision_recall_curve(y_val, winning_val_proba)
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
best_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[best_idx]

print(f"Optimal Decision Threshold: {optimal_threshold:.4f}")
print(f"Optimized Validation F1-Score: {f1_scores[best_idx]:.4f}")

# Re-evaluate winning model with the custom threshold
tuned_val_preds = (winning_val_proba >= optimal_threshold).astype(int)
print("\nClassification Report with Tuned Threshold:\n")
print(classification_report(y_val, tuned_val_preds, zero_division=0))


# =====================================================================
# CELL 9: RETRAIN ON 100% OF TRAINING DATA & PREDICT ON EXAM TEST SET
# =====================================================================
print("\n--- 9. RETRAINING ON 100% OF DATA & GENERATING PREDICTIONS ---")

# Retrain the winning pipeline on all available training data
winning_model.fit(X_full, y_full)

# Generate probabilities for the professor's test set
final_test_proba = winning_model.predict_proba(X_test_exam)[:, 1]

# Apply the optimized decision threshold
final_predictions = (final_test_proba >= optimal_threshold).astype(int)


# =====================================================================
# CELL 10: FINAL SUBMISSION OUTPUT
# =====================================================================
print("\n=========================================================")
print("FINAL EXAM PREDICTIONS ARRAY:")
print("=========================================================")
print(final_predictions)
print(f"Total rows predicted: {len(final_predictions)}")
print(f"Class distribution in predictions: {np.bincount(final_predictions)}")

# Optional: Save predictions to a CSV file if your exam environment requires file uploads
# pd.DataFrame({'prediction': final_predictions}).to_csv('exam_predictions.csv', index=False)