# Cell 1: Import Essential LibrariesPythonimport pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Cell 2: Load Your DatasetReplace 'your_dataset.csv' with the path to your actual dataset.Python# Load your raw dataset
df = pd.read_csv('your_dataset.csv')

# Inspect the first few rows
print(f"Dataset shape: {df.shape}")
display(df.head())

# Cell 3: Define Features (X) and Target (y)Separate your independent predictor variables ($X$) from your target variable ($y$).Python# MODIFY THESE based on your dataset columns:
target_column = 'target' 

X = df.drop(columns=[target_column])
y = df[target_column]

# Cell 4: Train / Test Split (CRUCIAL FIRST STEP)Always split your data before touching missing values, encoding, or scaling to prevent data leakage from the test set into your training set.PythonX_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,       # Reserves 20% of data for testing
    random_state=42,     # Ensures reproducibility
    stratify=y if y.dtype == 'O' or y.nunique() < 20 else None # Keeps class balance if classification
)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Cell 5: Automatically Identify Column TypesWe dynamically separate numerical and categorical columns so the preprocessor handles them correctly.Python# Identify numerical and categorical columns from the training set
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

print(f"Numerical features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# Cell 6: Build the Preprocessing PipelineThis pipeline ensures that statistics for missing values (like medians) and scalers (like means/variances) are calculated only on the training data and safely applied to the test data.Python# 1. Pipeline for Numerical Data: Impute missing values with median, then scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 2. Pipeline for Categorical Data: Fill missing with a constant 'missing' tag, then One-Hot Encode (dummy encoding)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first')) # drop='first' prevents dummy variable trap
])

# 3. Combine both into a single ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Cell 7: Fit and Transform the DataFit & Transform on the training data (X_train) so the preprocessor learns rules from training data only.Transform only on the test data (X_test) to prevent information leakage.Python# Fit on train and transform train
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test only
X_test_processed = preprocessor.transform(X_test)

print("Preprocessing complete successfully!")
print(f"Processed training features shape: {X_train_processed.shape}")
print(f"Processed testing features shape: {X_test_processed.shape}")

# Cell 8: (Optional) Convert Back to DataFrames for InspectionIf you want to view your processed arrays back into clean pandas DataFrames:Python# Retrieve encoded feature names (especially useful after OneHotEncoding)
try:
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    encoded_cat_features = cat_encoder.get_feature_names_out(categorical_features).tolist()
except AttributeError:
    encoded_cat_features = []

all_columns = numeric_features + encoded_cat_features

# Create clean DataFrames
X_train_df = pd.DataFrame(X_train_processed, columns=all_columns)
X_test_df = pd.DataFrame(X_test_processed, columns=all_columns)

display(X_train_df.head())