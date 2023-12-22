

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from datetime import datetime

output='Depth'
# input_file='DataAug/LPBF_Final_Dataset/LPBF_Dataset_Normalized_Depth.csv'

# output = 'Width'
input_file = f'D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Normalized_{output}.csv'

df = pd.read_csv(input_file)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Specify different kernels and their parameter grids
kernels = ['linear', 'poly', 'rbf']
C_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
degree_values = [2, 3, 4]
gamma_values = [0.0001, 0.001, 0.01, 0.1, 1, 10]

cv = KFold(n_splits=10, shuffle=True, random_state=42)

poly_list = []

for kernel in kernels:
    for C in C_values:
        if kernel == 'poly':
            for degree in degree_values:
                model = make_pipeline(StandardScaler(), SVR(kernel=kernel, C=C, degree=degree))

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
                    'Kernel': kernel,
                    'C': C,
                    'degree/gamma': degree,
                    'R-squared': corr,
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'RAE': rae,
                    'RRSE': rrse
                })

                scores = cross_val_score(model, X, y, cv=KFold(n_splits=10, shuffle=True, random_state=42),
                                         scoring='neg_mean_squared_error')
                print(f"Kernel: {kernel}, C: {C}, Degree: {degree}, Mean Squared Error: {-scores.mean()}")
        elif kernel == 'rbf':
            for gamma in gamma_values:
                model = make_pipeline(StandardScaler(), SVR(kernel=kernel, C=C, gamma=gamma))

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
                    'Kernel': kernel,
                    'C': C,
                    'degree/gamma': gamma,
                    'R-squared': corr,
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'RAE': rae,
                    'RRSE': rrse
                })

                scores = cross_val_score(model, X, y, cv=KFold(n_splits=10, shuffle=True, random_state=42),
                                         scoring='neg_mean_squared_error')
                print(f"Kernel: {kernel}, C: {C}, Gamma: {gamma}, Mean Squared Error: {-scores.mean()}")
        else:
            model = make_pipeline(StandardScaler(), SVR(kernel=kernel, C=C))
            scores = cross_val_score(model, X, y, cv=KFold(n_splits=10, shuffle=True, random_state=42),
                                     scoring='neg_mean_squared_error')
            print(f"Kernel: {kernel}, C: {C}, Mean Squared Error: {-scores.mean()}")

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
                'Kernel': kernel,
                'C': C,
                'degree/gamma': 0,
                'R-squared': corr,
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'RAE': rae,
                'RRSE': rrse
            })

poly_results = pd.DataFrame(poly_list)
# Save results to a CSV file

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
filename = f'LPBF_HP_SupportVectorRegression_results_{output}_{timestamp}.csv'

poly_results.to_csv(filename, index=False)