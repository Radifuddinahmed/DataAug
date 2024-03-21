import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import make_regression
from datetime import datetime
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import RationalQuadratic

output = 'Width'
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
base_estimator_values = [None, SVR(), DecisionTreeRegressor(), LinearRegression()] #default=None
n_estimators_values = [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] #default=10
max_sample_values = [x / 10 for x in range(1, 11)] #default=1.0
max_feature_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.90, 0.92, 0.95, 1.0] #default=1.0


# # Specify different hyperparameter combinations
# base_estimator_values = [None, SVR(), DecisionTreeRegressor(), LinearRegression()] #default=None
# n_estimators_values = [1, 5, 10, 30, 50, 100, 200] #default=10
# max_sample_values = [0.1, 0.5, 1, 2, 3, 4] #default=1.0
# max_feature_values = [0.1, 0.5, 1, 2, 3, 4] #default=1.0


poly_list = []
cv = KFold(n_splits=10, shuffle=True, random_state=42)
# Perform 10-fold cross-validation for each combination of hyperparameters
for base_estimators in base_estimator_values:
    for n_estimators in n_estimators_values:
        for max_samples in max_sample_values:
            for max_features in max_feature_values:
                model = BaggingRegressor(
                                        estimator=base_estimators,
                                        n_estimators=n_estimators,
                                        max_samples=max_samples,
                                        max_features=max_features,
                                        bootstrap=True,
                                        bootstrap_features=False,
                                        oob_score=False,
                                        warm_start=False,
                                        n_jobs=None,
                                        random_state=42,
                                        verbose=0,
                                        base_estimator="deprecated",
                                         )

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
                    'base_estimator': base_estimators,
                    'n_estimators': n_estimators,
                    'max_samples': max_samples,
                    'max_features': max_features,
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
                print(
                    f"base_estimator: {base_estimators}, n_estimators: {n_estimators}, max_samples: {max_samples}, max_features: {max_features}, Mean Absolute Error: {mae}")

poly_results = pd.DataFrame(poly_list)
# Save results to a CSV file

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
filename = f'LPBF_HP_Bagging_results_{output}_{timestamp}.csv'

poly_results.to_csv(filename, index=False)
