# Cell 1: Import Evaluation Metrics & Plotting Tools
# Python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,               # Cross-Entropy Loss
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    classification_report
)
# Cell 2: Compute and Print All Evaluation Metrics
# Python
# Assuming you have:
# y_test: actual ground truth labels (0 or 1)
# y_pred = model.predict(X_test)
# y_proba = model.predict_proba(X_test)[:, 1]

# 1. Accuracy
acc = accuracy_score(y_test, y_pred)

# 2. Precision
precision = precision_score(y_test, y_pred)

# 3. Recall
recall = recall_score(y_test, y_pred)

# 4. F1-Score
f1 = f1_score(y_test, y_pred)

# 5. Cross-Entropy Loss (Log Loss)
cross_entropy = log_loss(y_test, y_proba)

# 6. ROC AUC Score
roc_auc = roc_auc_score(y_test, y_proba)

# 7. Cohen's Kappa
kappa = cohen_kappa_score(y_test, y_pred)

# Print Summary Table
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

# Detailed per-class breakdown
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
# Cell 3: Plot Confusion Matrix (Stacked)
# Python
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
disp.plot(cmap='Blues', colorbar=False, ax=plt.gca())
plt.title("Confusion Matrix", fontsize=14)
plt.grid(False)
plt.show()
# Cell 4: Plot ROC AUC Curve (Stacked)
# Python
plt.figure(figsize=(6, 5))
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR / Recall)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC)', fontsize=14)
plt.legend(loc="lower right")
plt.show()