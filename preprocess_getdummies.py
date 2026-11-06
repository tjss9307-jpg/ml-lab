# Cell 1: Import Libraries
Python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Cell 2: Load Your Dataset
Python
# Load your dataset (replace with your file path)
df = pd.read_csv('your_dataset.csv')

# Define your target column and features
target_column = 'target' 
X = df.drop(columns=[target_column])
y = df[target_column]

# Cell 3: Train / Test Split (CRUCIAL FIRST STEP)
Always split your data before handling missing values or dummy encoding.

Python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.25, 
    random_state=42
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Cell 4: Handle Missing Values (Imputation)
# We handle missing values independently for numerical and categorical columns on the training set, and apply the learned values to the test set.

Python
# 1. Numerical Missing Values (Fill with Train Median)
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    train_median = X_train[col].median()
    X_train[col] = X_train[col].fillna(train_median)
    X_test[col] = X_test[col].fillna(train_median) # Use train median on test!

# 2. Categorical Missing Values (Fill with 'missing' string)
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns

for col in cat_cols:
    X_train[col] = X_train[col].fillna('missing')
    X_test[col] = X_test[col].fillna('missing')

# Cell 5: Apply pd.get_dummies() & Align Columns
Important Note: When using pd.get_dummies() on train and test separately, the test set might end up missing certain rare categories that were only present in the training set (or vice versa). Using .align() ensures both datasets have the exact same columns.

Python
# Apply get_dummies with drop_first=True to avoid the dummy variable trap
X_train_encoded = pd.get_dummies(X_train, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, drop_first=True)

# Align the test set columns to match the training set columns perfectly
X_train_encoded, X_test_encoded = X_train_encoded.align(
    X_test_encoded, 
    join='left', 
    axis=1, 
    fill_value=0
)

print(f"Processed Train Shape: {X_train_encoded.shape}")
print(f"Processed Test Shape: {X_test_encoded.shape}")

# Cell 6: Verify and Ready for Modeling
Python
# Check the first few columns of your successfully preprocessed training data
display(X_train_encoded.head())