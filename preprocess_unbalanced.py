# Cell 1: Import Libraries
# Python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Cell 2: Load Data and Separate Features & Target
# Python
# Replace with your dataset file path
df = pd.read_csv('your_dataset.csv')

target_col = 'target'  # Set your target column name
X = df.drop(columns=[target_col])
y = df[target_col]

# Check class imbalance
print("Class distribution:")
print(y.value_counts(normalize=True) * 100)
# 
Cell 3: Stratified Train / Test Split (CRUCIAL)
Setting stratify=y guarantees that both splits preserve the same percentage of the minority class.

Python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.25, 
    random_state=42, 
    stratify=y  # Essential for imbalanced datasets
)

print(f"X_train shape: {X_train.shape}, Class 1 count: {sum(y_train == 1)}")
print(f"X_test shape:  {X_test.shape}, Class 1 count: {sum(y_test == 1)}")

# Cell 4: Handle Missing Values (Separately on Train & Test)

Python
# Separate numerical and categorical columns
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns

# 1. Impute numerical missing values with train median
for col in num_cols:
    train_median = X_train[col].median()
    X_train[col] = X_train[col].fillna(train_median)
    X_test[col] = X_test[col].fillna(train_median)

# 2. Impute categorical missing values with 'missing'
for col in cat_cols:
    X_train[col] = X_train[col].fillna('missing')
    X_test[col] = X_test[col].fillna('missing')

# Cell 5: Dummy Encoding (pd.get_dummies) & Column Alignment
Python
# Apply get_dummies
X_train_encoded = pd.get_dummies(X_train, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, drop_first=True)

# Align columns to guarantee both sets have identical dummy feature names
X_train_encoded, X_test_encoded = X_train_encoded.align(
    X_test_encoded, 
    join='left', 
    axis=1, 
    fill_value=0
)

# Cell 6: Feature Scaling
Fit the scaler only on the training data, then transform both training and testing datasets.

Python
scaler = StandardScaler()

# Fit on train, transform both
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_encoded), 
    columns=X_train_encoded.columns, 
    index=X_train_encoded.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_encoded), 
    columns=X_test_encoded.columns, 
    index=X_test_encoded.index
)

# Cell 7: Balance the Minority Class using SMOTE (Train Set ONLY)
Applying SMOTE creates synthetic samples for the minority class in X_train without touching X_test.

Python
# Apply SMOTE strictly on training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print("Before SMOTE training labels:", y_train.value_counts().to_dict())
print("After SMOTE training labels: ", pd.Series(y_train_balanced).value_counts().to_dict())

# Cell 8: Train Model and Evaluate with Imbalance-Aware Metrics
For imbalanced classification, avoid evaluating purely on raw accuracy. Use Precision, Recall, F1-Score, and ROC-AUC.

Python
# Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_balanced, y_train_balanced)

# Predict on the untouched test set
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Display evaluation metrics
print("--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")