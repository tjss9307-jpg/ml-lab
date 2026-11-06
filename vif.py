import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- STEP A: Prepare your data (Must be numerical features only, no text/strings) ---
# Example DataFrame: X_train
# Note: It's best practice to drop or exclude the intercept column if your data already has one.

# --- STEP B: Create a function to calculate VIF ---
def calculate_vif(X):
    # Create a DataFrame to store VIF values
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    
    # Calculate VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    # Sort from highest to lowest VIF
    vif_data = vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)
    
    return vif_data

# --- STEP C: Run the function on your numerical training features ---
# (Make sure X_train contains only numeric/scaled columns and no missing values/NaNs)
# vif_results = calculate_vif(X_train)
# display(vif_results)