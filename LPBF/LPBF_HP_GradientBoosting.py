import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import make_regression
from datetime import datetime

output = 'Depth'
input_file = f'D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Normalized_{output}.csv'

# df = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Normalized_Width.csv')
# X = df[['Laser Power [W]', 'Scanning Speed [mm/s]', 'Layer Thickness [um]', 'Spot Size [um]', 'Porosity [%]']].values
# y = df['Max Melt Pool Width [um]'].values

# output='width'
# input_file='DataAug/LPBF_Final_Dataset/LPBF_Dataset_Normalized_Width.csv'

df = pd.read_csv(input_file)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Specify different hyperparameter combinations
n_estimators_values = [10, 50, 100, 200]
max_depth_values = [None, 10, 20, 30]
min_samples_split_values = [2, 5, 10]
min_samples_leaf_values = [1, 2, 4]


poly_list = []
cv = KFold(n_splits=10, shuffle=True, random_state=42)
# Perform 10-fold cross-validation for each combination of hyperparameters
for n_estimators in n_estimators_values:
    for max_depth in max_depth_values:
        for min_samples_split in min_samples_split_values:
            for min_samples_leaf in min_samples_leaf_values:
                model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                              min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                              random_state=42)

                mse_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
                mae_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
                r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

                # Calculate metrics
                mse = mse_scores.mean()
                mae = mae_scores.mean()
                corr = r2_scores.mean()

                # Calculate additional metrics
                rmse = np.sqrt(mse)
                rae = mae / np.mean(np.abs(y - np.mean(y)))
                rrse = rmse / np.sqrt(np.mean((y - np.mean(y)) ** 2))

                poly_list.append({
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'R-squared': corr,
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'RAE': rae,
                    'RRSE': rrse
                })

                scores = cross_val_score(model, X, y, cv=cv,
                                         scoring='neg_mean_squared_error')

                # Display the results
                print(f"n_estimators: {n_estimators}, max_depth: {max_depth}, min_samples_split: {min_samples_split}, "
                      f"min_samples_leaf: {min_samples_leaf}, Mean Squared Error: {-scores.mean()}")

poly_results = pd.DataFrame(poly_list)
# Save results to a CSV file

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
filename = f'LPBF_HP_GradientBoosting_results_{output}_{timestamp}.csv'

poly_results.to_csv(filename, index=False)