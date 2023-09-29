# Import necessary libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
)

# Load the dataset as an example
df = pd.read_csv("data.csv")
X = df[['Tool Rotational Speed (RPM)', 'Translational Speed (mm/min)', 'Axial Force (KN)']].values
y = df['Ultimate Tensile Trength (MPa)'].values



# Create a list of ensemble regressors
ensemble_methods = [
    ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('AdaBoost', AdaBoostRegressor(n_estimators=100, random_state=42)),
    ('Bagging', BaggingRegressor(n_estimators=100, random_state=42)),
    ('Extra Trees', ExtraTreesRegressor(n_estimators=100, random_state=42)),
]
k=10
# Compare ensemble methods using cross-validation
for name, model in ensemble_methods:
    scores = cross_val_score(model, X, y, cv=k, scoring='neg_mean_squared_error')
    mse_scores = -scores  # Convert negative MSE scores to positive
    r2_scores = cross_val_score(model, X, y, cv=k, scoring='r2')
    
    print(f"{name} - Mean MSE: {np.mean(mse_scores):.2f} (+/- {np.std(mse_scores):.2f})")
    print(f"{name} - Mean R-squared (R2): {np.mean(r2_scores):.2f} (+/- {np.std(r2_scores):.2f})")
