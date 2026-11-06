import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve, 
    f1_score, accuracy_score, confusion_matrix
)

# ==========================================
# STEP 1: LOAD DATA
# ==========================================
# train.csv has features + target labels
# test.csv is the professor's unseen file with NO labels for grading
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

target_col = 'target' # <-- CHANGE THIS to your actual target column name

X_full = train_df.drop(columns=[target_col])
y_full = train_df[target_col]
X_test_exam = test_df.copy() # Unseen grading set

# ==========================================
# STEP 2: INTERNAL TRAIN / VALIDATION SPLIT
# ==========================================
# This lets you check your own metrics, compare models, and tune thresholds!
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)


# ==========================================
# STEP 3: PREPROCESSING (Impute, Encode, Align, Scale)
# ==========================================
# 1. Impute missing values (using stats strictly calculated from X_train)
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns

for col in num_cols:
    median_val = X_train[col].median()
    X_train[col] = X_train[col].fillna(median_val)
    X_val[col] = X_val[col].fillna(median_val)
    X_test_exam[col] = X_test_exam[col].fillna(median_val)

for col in cat_cols:
    X_train[col] = X_train[col].fillna('missing')
    X_val[col] = X_val[col].fillna('missing')
    X_test_exam[col] = X_test_exam[col].fillna('missing')

# 2. Dummy Encoding & Column Alignment (Ensures shapes match across all sets)
X_train_enc = pd.get_dummies(X_train, drop_first=True)
X_val_enc = pd.get_dummies(X_val, drop_first=True)
X_test_enc = pd.get_dummies(X_test_exam, drop_first=True)

X_train_enc, X_val_enc = X_train_enc.align(X_val_enc, join='left', axis=1, fill_value=0)
X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join='left', axis=1, fill_value=0)

# 3. Standardization (Fit scaler on X_train_enc ONLY)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_enc)
X_val_scaled = scaler.transform(X_val_enc)
X_test_scaled = scaler.transform(X_test_enc)


# ==========================================
# STEP 4: MODEL 1 - LOGISTIC REGRESSION (GridSearchCV)
# ==========================================
print("--- Training & Tuning Logistic Regression ---")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_param_grid = {'C': [0.01, 0.1, 1.0, 10.0]}

lr_grid = GridSearchCV(estimator=lr_model, param_grid=lr_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
lr_grid.fit(X_train_scaled, y_train)

best_lr = lr_grid.best_estimator_
print(f"Best LR Params: {lr_grid.best_params_} | Best CV Score: {lr_grid.best_score_:.4f}")


# ==========================================
# STEP 5: MODEL 2 - SUPPORT VECTOR MACHINE / SVM (GridSearchCV)
# ==========================================
print("\n--- Training & Tuning SVM ---")
svm_model = SVC(probability=True, random_state=42) # probability=True required for predict_proba!
svm_param_grid = {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']}

svm_grid = GridSearchCV(estimator=svm_model, param_grid=svm_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
svm_grid.fit(X_train_scaled, y_train)

best_svm = svm_grid.best_estimator_
print(f"Best SVM Params: {svm_grid.best_params_} | Best CV Score: {svm_grid.best_score_:.4f}")


# ==========================================
# STEP 6: COMPARE MODELS ON VALIDATION SET
# ==========================================
# Evaluate both on your internal validation set to see metrics
lr_val_acc = accuracy_score(y_val, best_lr.predict(X_val_scaled))
svm_val_acc = accuracy_score(y_val, best_svm.predict(X_val_scaled))

print(f"\nValidation Accuracy - Logistic Regression: {lr_val_acc:.4f}")
print(f"Validation Accuracy - SVM: {svm_val_acc:.4f}")

# Automatically choose the winning model based on validation performance
if lr_val_acc >= svm_val_acc:
    winning_model = best_lr
    print("\n>>> Winner selected for final predictions: Logistic Regression")
else:
    winning_model = best_svm
    print("\n>>> Winner selected for final predictions: SVM")


# ==========================================
# STEP 7: THRESHOLD TUNING (Crucial for Imbalanced Data)
# ==========================================
# Get probabilities on the validation set using the winning model
val_proba = winning_model.predict_proba(X_val_scaled)[:, 1]

# Use Precision-Recall curve to find the optimal decision threshold maximizing F1-score
precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
best_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[best_idx]

print(f"Optimal Decision Threshold found: {optimal_threshold:.4f}")
print(f"Validation F1-Score with Tuned Threshold: {f1_scores[best_idx]:.4f}")

# Print final classification report for validation set
val_preds_tuned = (val_proba >= optimal_threshold).astype(int)
print("\nValidation Classification Report (Tuned Threshold):\n", classification_report(y_val, val_preds_tuned))


# ==========================================
# STEP 8: RETRAIN WINNING MODEL ON 100% OF DATA & PREDICT ON EXAM TEST SET
# ==========================================
# For maximum performance, retrain the winning model architecture using the 
# best hyperparameters found during CV on ALL available training data (X_full_scaled)

# 1. Preprocess full dataset for final fit
X_full_enc = pd.get_dummies(X_full, drop_first=True)
X_full_enc, X_test_final_enc = X_full_enc.align(X_test_enc, join='left', axis=1, fill_value=0)

final_scaler = StandardScaler()
X_full_scaled = final_scaler.fit_transform(X_full_enc)
X_test_final_scaled = final_scaler.transform(X_test_final_enc)

# 2. Retrain winning model on 100% of training data
winning_model.fit(X_full_scaled, y_full)

# 3. Generate final probabilities and apply your tuned threshold
final_test_proba = winning_model.predict_proba(X_test_scaled)[:, 1]
final_predictions = (final_test_proba >= optimal_threshold).astype(int)

# ==========================================
# STEP 9: FINAL SUBMISSION OUTPUT
# ==========================================
print("\n==========================================")
print("FINAL PREDICTIONS FOR EXAM TEST SET READY:")
print("==========================================")
print(final_predictions)
print(f"Total predictions generated: {len(final_predictions)}")

# (Optional: Save to CSV if your lab requires it)
# pd.DataFrame({'prediction': final_predictions}).to_csv('my_predictions.csv', index=False)