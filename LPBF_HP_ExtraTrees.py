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

n_estimators_values = [10, 20, 50, 100, 200, 500, 1000, 1200, 1500, 1800, 1900, 2000, 2100] #default 100
max_depth_values = [1, 2, 5, 8, 13, 21, 34, 53, 54, 55, 89, None] #default None
min_samples_split_values = [ 2, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377] #default 2
min_samples_leaf_values = [1, 2, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]  #default 1

# n_estimators_values = [100, 200, 300] #default 100
# max_depth_values = [None, 3, 5, 7, 10] #default None
# min_samples_split_values = [ 2, 3] #default 2
# min_samples_leaf_values = [1, 2, 3, 4]  #default 1

poly_list = []
cv = KFold(n_splits=10, shuffle=True, random_state=42)
# Perform 10-fold cross-validation for each combination of hyperparameters
for n_estimators in n_estimators_values:
    for max_depth in max_depth_values:
        for min_samples_split in min_samples_split_values:
            for min_samples_leaf in min_samples_leaf_values:
                model = ExtraTreesRegressor(n_estimators= n_estimators,
                                            max_depth= max_depth,
                                         min_samples_split= min_samples_split,
                                            min_samples_leaf= min_samples_leaf,
                                            random_state=42,
                                            # n_estimators=100,
                                            # *,
                                            # criterion="squared_error",
                                            # max_depth=None,
                                            # min_samples_split=2,
                                            # min_samples_leaf=1,
                                            # min_weight_fraction_leaf=0.0,
                                            # max_features=1.0,
                                            # max_leaf_nodes=None,
                                            # min_impurity_decrease=0.0,
                                            # bootstrap=False,
                                            # oob_score=False,
                                            # n_jobs=None,
                                            # random_state=42,
                                            # verbose=0,
                                            # warm_start=False,
                                            # ccp_alpha=0.0,
                                            # max_samples=None,
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
                print(
                    f"n_estimators: {n_estimators}, max_depth: {max_depth}, min_samples_split: {min_samples_split}, min_samples_leaf: {min_samples_leaf}, Mean Absolute Error: {mae}")

poly_results = pd.DataFrame(poly_list)
# Save results to a CSV file

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
filename = f'LPBF_HP_ExtraTrees_results_{output}_{timestamp}.csv'

poly_results.to_csv(filename, index=False)
