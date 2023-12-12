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

#!###############################################!#
#!                                               !#
#!####################  Width      ##############!#
#!                                               !#
#!###############################################!#


# # Load the dataset as an example
# df = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Normalized_Width.csv')
# X = df[['Laser Power [W]', 'Scanning Speed [mm/s]', 'Layer Thickness [um]', 'Spot Size [um]', 'Porosity [%]']].values
# y = df['Max Melt Pool Width [um]'].values
#
# poly = PolynomialFeatures(degree=7, include_bias=False)
# poly_features = poly.fit_transform(X)
# poly_reg_model = LinearRegression()
#
#
# c1 = []
# c2 = []
# c3 = []
# c4 = []
# c5 = []
# c6 = []
# opt_data = {'Model': c1,
#             'Correlation Coefficient': c2,
#             'MAE': c3,
#             'RMSE': c4,
#             'RAE': c5,
#             'RRSE':c6 }
# opt = pd.DataFrame(opt_data)
#
#
# # Create a list of ensemble regressors
# ensemble_methods = [
#     ('Gaussian Process', GaussianProcessRegressor(kernel=RationalQuadratic(alpha=1, length_scale=1),random_state=42).fit(X, y)),
#     ('Linear Regression', LinearRegression()),
#     ('Polynomial Regression', poly_reg_model.fit(poly_features, y)),
#     ('Support Vector Regression', SVR(kernel='linear', C=1000, gamma=1e-05)),
#     ('KNN', KNeighborsRegressor(n_neighbors=3)),
#     ('Multi Layer Perception', MLPRegressor(activation='identity',batch_size=32,hidden_layer_sizes=4, learning_rate='invscaling',max_iter= 500,alpha=0.001,random_state=42,early_stopping=False)),
#     ('Random Forest', RandomForestRegressor(n_estimators=50,max_depth=10, min_samples_split=2, random_state=42)),
#     ('Gradient Boosting', GradientBoostingRegressor(n_estimators=200,learning_rate=0.1, max_depth=5, min_samples_split=2 , random_state=42)),
#     ('AdaBoost', AdaBoostRegressor(base_estimator=LinearRegression(),learning_rate=.001,n_estimators=10, random_state=42)),
#     ('Bagging', BaggingRegressor(base_estimator=DecisionTreeRegressor(),n_estimators=10, random_state=42)),
#     ('Extra Trees', ExtraTreesRegressor(max_depth=10,min_samples_split=5,n_estimators=100, random_state=42)),
# ]
# k = 10
# # Compare ensemble methods using cross-validation
# for name, model in ensemble_methods:
#     scores = cross_val_score(model, X, y, cv=k, scoring='neg_mean_squared_error')
#     predicted = cross_val_predict(model, X, y, cv=k)
#     correlation_coefficient = r2_score(y, predicted)
#     mse_scores = -scores  # Convert negative MSE scores to positive
#     rmse_scores = np.sqrt(mse_scores)
#     r2_scores = cross_val_score(model, X, y, cv=k, scoring='r2')
#     # Relative Absolute Error
#     absolute_errors = np.abs(y - predicted)
#     mean_absolute_error = np.mean(absolute_errors)
#     mean_true_values = np.mean(y)
#     relative_absolute_error = mean_absolute_error / mean_true_values
#     # Calculate the squared errors
#     squared_errors = np.square(y - predicted)
#     mean_squared_error = np.mean(squared_errors)
#     mean_true_values = np.mean(y)
#     root_relative_squared_error = np.sqrt(mean_squared_error) / mean_true_values
#
#
#     print(f"{name} - Correlation Coefficient: {correlation_coefficient:.2f} (+/- {np.std(correlation_coefficient):.2f})")
#     print(f"{name} - Mean MSE: {np.mean(mse_scores):.2f} (+/- {np.std(mse_scores):.2f})")
#     print(f"{name} - Mean Absolute Error: {np.mean(mse_scores):.2f} (+/- {np.std(mse_scores):.2f})")
#     print(f"{name} - Root Mean Squared Error: {np.mean(rmse_scores):.2f} (+/- {np.std(rmse_scores):.2f})")
#     print(f"{name} - Mean R-squared (R2): {np.mean(r2_scores):.2f} (+/- {np.std(r2_scores):.2f})")
#     print(f"{name} - Relative Absolute Error: {relative_absolute_error:.2f} (+/- {np.std(relative_absolute_error):.2f})")
#     print(f"{name} - Root Relative Squared Error (RRSE): {root_relative_squared_error:.2f} (+/- {np.std(relative_absolute_error):.2f})")
#
#     opt = pd.concat([opt, pd.DataFrame.from_records([{'Model': f"{name}",
#                                                       'Correlation Coefficient': correlation_coefficient,
#                                                       'MAE': np.mean(mse_scores),
#                                                       'RMSE': np.mean(rmse_scores),
#                                                       'RAE': relative_absolute_error,
#                                                       'RRSE': root_relative_squared_error}])])
#
#
#
#
# opt.to_csv('All_ML_models_Results_v3_HpTuned_Width.csv',mode='w', index=False)
#
#
# #!###############################################!#
# #!                                               !#
# #!####################  Depth ###################!#
# #!                                               !#
# #!###############################################!#
#
#
# # Load the dataset as an example
# df = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Normalized_Depth.csv')
# X = df[['Laser Power [W]', 'Scanning Speed [mm/s]', 'Layer Thickness [um]', 'Spot Size [um]', 'Porosity [%]']].values
# y = df['Max Melt Pool Depth [um]'].values
#
# poly = PolynomialFeatures(degree=7, include_bias=False)
# poly_features = poly.fit_transform(X)
# poly_reg_model = LinearRegression()
#
#
# c1 = []
# c2 = []
# c3 = []
# c4 = []
# c5 = []
# c6 = []
# opt_data = {'Model': c1,
#             'Correlation Coefficient': c2,
#             'MAE': c3,
#             'RMSE': c4,
#             'RAE': c5,
#             'RRSE':c6 }
# opt = pd.DataFrame(opt_data)
#
# # Create a list of ensemble regressors
# ensemble_methods = [
#     ('Gaussian Process', GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(),random_state=42).fit(X, y)),
#     ('Linear Regression', LinearRegression()),
#     ('Polynomial Regression', poly_reg_model.fit(poly_features, y)),
#     ('Support Vector Regression', SVR(kernel='sigmoid', C=1000, gamma=.001)),
#     ('KNN', KNeighborsRegressor(n_neighbors=4)),
#     ('Multi Layer Perception', MLPRegressor(activation='relu',batch_size=128,hidden_layer_sizes=4, learning_rate='constant',alpha=0.001,random_state=42,early_stopping=False)),
#     ('Random Forest', RandomForestRegressor(n_estimators=50,max_depth=10, min_samples_split=2, random_state=42)),
#     ('Gradient Boosting', GradientBoostingRegressor(n_estimators=200,learning_rate=1, max_depth=3, min_samples_split=10 , random_state=42)),
#     ('AdaBoost', AdaBoostRegressor(base_estimator= DecisionTreeRegressor(max_depth=5),learning_rate=.001,n_estimators=10, random_state=42)),
#     ('Bagging', BaggingRegressor(base_estimator= DecisionTreeRegressor(),n_estimators=200, random_state=42)),
#     ('Extra Trees', ExtraTreesRegressor(max_depth=10,min_samples_split=5,n_estimators=200, random_state=42)),
# ]
# k = 10
# # Compare ensemble methods using cross-validation
# for name, model in ensemble_methods:
#     scores = cross_val_score(model, X, y, cv=k, scoring='neg_mean_squared_error')
#     predicted = cross_val_predict(model, X, y, cv=k)
#     correlation_coefficient = r2_score(y, predicted)
#     mse_scores = -scores  # Convert negative MSE scores to positive
#     rmse_scores = np.sqrt(mse_scores)
#     r2_scores = cross_val_score(model, X, y, cv=k, scoring='r2')
#     # Relative Absolute Error
#     absolute_errors = np.abs(y - predicted)
#     mean_absolute_error = np.mean(absolute_errors)
#     mean_true_values = np.mean(y)
#     relative_absolute_error = mean_absolute_error / mean_true_values
#     # Calculate the squared errors
#     squared_errors = np.square(y - predicted)
#     mean_squared_error = np.mean(squared_errors)
#     mean_true_values = np.mean(y)
#     root_relative_squared_error = np.sqrt(mean_squared_error) / mean_true_values
#
#
#     print(f"{name} - Correlation Coefficient: {correlation_coefficient:.2f} (+/- {np.std(correlation_coefficient):.2f})")
#     print(f"{name} - Mean MSE: {np.mean(mse_scores):.2f} (+/- {np.std(mse_scores):.2f})")
#     print(f"{name} - Mean Absolute Error: {np.mean(mse_scores):.2f} (+/- {np.std(mse_scores):.2f})")
#     print(f"{name} - Root Mean Squared Error: {np.mean(rmse_scores):.2f} (+/- {np.std(rmse_scores):.2f})")
#     print(f"{name} - Mean R-squared (R2): {np.mean(r2_scores):.2f} (+/- {np.std(r2_scores):.2f})")
#     print(f"{name} - Relative Absolute Error: {relative_absolute_error:.2f} (+/- {np.std(relative_absolute_error):.2f})")
#     print(f"{name} - Root Relative Squared Error (RRSE): {root_relative_squared_error:.2f} (+/- {np.std(relative_absolute_error):.2f})")
#
#     opt = pd.concat([opt, pd.DataFrame.from_records([{'Model': f"{name}",
#                                                       'Correlation Coefficient': correlation_coefficient,
#                                                       'MAE': np.mean(mse_scores),
#                                                       'RMSE': np.mean(rmse_scores),
#                                                       'RAE': relative_absolute_error,
#                                                       'RRSE': root_relative_squared_error}])])
#
#
#
#
# opt.to_csv('All_ML_models_Results_v3_HpTuned_Depth.csv',mode='w', index=False)


#!###############################################!#
#!                                               !#
#!####################  Width Poly ##############!#
#!             with original values              !#
#!###############################################!#


# Load the dataset as an example
# df = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_polyExp(6features)Normalized.csv')
# X = df[['Laser Power [W]', 'Scanning Speed [mm/s]', 'Layer Thickness [um]', 'Spot Size [um]', 'Porosity [%]','LPxLT','LPxSS','LPxP','SSpxLT','SSpxSS','SSpxP']].values
# y = df['Max Melt Pool Width [um]'].values

df = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_polyExp(12features)Normalized.csv')
X = df[['Laser Power [W]', 'Scanning Speed [mm/s]', 'Layer Thickness [um]', 'Spot Size [um]', 'Porosity [%]','LPxLT','LPxSS','LPxP','SSpxLT','SSpxSS','SSpxP','LP2xLT','LP2xSS','LP2xP','SSp2xLT','SSp2xSS','SSp2xP']].values
y = df['Max Melt Pool Width [um]'].values

#,'SSpxP'

poly = PolynomialFeatures(degree=7, include_bias=False)
poly_features = poly.fit_transform(X)
poly_reg_model = LinearRegression()


c1 = []
c2 = []
c3 = []
c4 = []
c5 = []
c6 = []
opt_data = {'Model': c1,
            'Correlation Coefficient': c2,
            'MAE': c3,
            'RMSE': c4,
            'RAE': c5,
            'RRSE':c6 }
opt = pd.DataFrame(opt_data)


ensemble_methods = [
    ('Gaussian Process', GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(), random_state=42).fit(X, y)),
    ('Linear Regression', LinearRegression()),
    ('Polynomial Regression', poly_reg_model.fit(poly_features, y)),
    ('Support Vector Regression', SVR(kernel='linear', C=.5)),
    ('KNN', KNeighborsRegressor(n_neighbors=2)),
    ('Multi Layer Perception',
     MLPRegressor(activation='relu', hidden_layer_sizes=(10, 100), alpha=0.001, random_state=42, early_stopping=False)),
    ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('AdaBoost', AdaBoostRegressor(n_estimators=100, random_state=42)),
    ('Bagging', BaggingRegressor(n_estimators=100, random_state=42)),
    ('Extra Trees', ExtraTreesRegressor(n_estimators=100, random_state=42)),
]
k = 10
# Compare ensemble methods using cross-validation
for name, model in ensemble_methods:
    scores = cross_val_score(model, X, y, cv=k, scoring='neg_mean_squared_error')
    predicted = cross_val_predict(model, X, y, cv=k)
    correlation_coefficient = r2_score(y, predicted)
    mse_scores = -scores  # Convert negative MSE scores to positive
    rmse_scores = np.sqrt(mse_scores)
    r2_scores = cross_val_score(model, X, y, cv=k, scoring='r2')
    # Relative Absolute Error
    absolute_errors = np.abs(y - predicted)
    mean_absolute_error = np.mean(absolute_errors)
    mean_true_values = np.mean(y)
    relative_absolute_error = mean_absolute_error / mean_true_values
    # Calculate the squared errors
    squared_errors = np.square(y - predicted)
    mean_squared_error = np.mean(squared_errors)
    mean_true_values = np.mean(y)
    root_relative_squared_error = np.sqrt(mean_squared_error) / mean_true_values


    print(f"{name} - Correlation Coefficient: {correlation_coefficient:.2f} (+/- {np.std(correlation_coefficient):.2f})")
    print(f"{name} - Mean MSE: {np.mean(mse_scores):.2f} (+/- {np.std(mse_scores):.2f})")
    print(f"{name} - Mean Absolute Error: {np.mean(mse_scores):.2f} (+/- {np.std(mse_scores):.2f})")
    print(f"{name} - Root Mean Squared Error: {np.mean(rmse_scores):.2f} (+/- {np.std(rmse_scores):.2f})")
    print(f"{name} - Mean R-squared (R2): {np.mean(r2_scores):.2f} (+/- {np.std(r2_scores):.2f})")
    print(f"{name} - Relative Absolute Error: {relative_absolute_error:.2f} (+/- {np.std(relative_absolute_error):.2f})")
    print(f"{name} - Root Relative Squared Error (RRSE): {root_relative_squared_error:.2f} (+/- {np.std(relative_absolute_error):.2f})")

    opt = pd.concat([opt, pd.DataFrame.from_records([{'Model': f"{name}",
                                                      'Correlation Coefficient': correlation_coefficient,
                                                      'MAE': np.mean(mse_scores),
                                                      'RMSE': np.mean(rmse_scores),
                                                      'RAE': relative_absolute_error,
                                                      'RRSE': root_relative_squared_error}])])




# opt.to_csv('All_ML_models_Results_v3_Poly_Width(6features).csv',mode='w', index=False)
opt.to_csv('All_ML_models_Results_v3_Poly_Width(12features).csv',mode='w', index=False)


#!###############################################!#
#!                                               !#
#!####################  Depth Poly ##############!#
#!                                               !#
#!###############################################!#


# Load the dataset as an example
# df = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_polyExp(6features)Normalized.csv')
# X = df[['Laser Power [W]', 'Scanning Speed [mm/s]', 'Layer Thickness [um]', 'Spot Size [um]', 'Porosity [%]','LPxLT','LPxSS','LPxP','SSpxLT','SSpxSS','SSpxP']].values
# y = df['Max Melt Pool Depth [um]'].values

df = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_polyExp(12features)Normalized.csv')
X = df[['Laser Power [W]', 'Scanning Speed [mm/s]', 'Layer Thickness [um]', 'Spot Size [um]', 'Porosity [%]','LPxLT','LPxSS','LPxP','SSpxLT','SSpxSS','SSpxP','LP2xLT','LP2xSS','LP2xP','SSp2xLT','SSp2xSS','SSp2xP']].values
y = df['Max Melt Pool Depth [um]'].values

poly = PolynomialFeatures(degree=7, include_bias=False)
poly_features = poly.fit_transform(X)
poly_reg_model = LinearRegression()

c1 = []
c2 = []
c3 = []
c4 = []
c5 = []
c6 = []
opt_data = {'Model': c1,
            'Correlation Coefficient': c2,
            'MAE': c3,
            'RMSE': c4,
            'RAE': c5,
            'RRSE':c6 }
opt = pd.DataFrame(opt_data)

# Create a list of ensemble regressors
ensemble_methods = [
    ('Gaussian Process', GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(), random_state=42).fit(X, y)),
    ('Linear Regression', LinearRegression()),
    ('Polynomial Regression', poly_reg_model.fit(poly_features, y)),
    ('Support Vector Regression', SVR(kernel='linear', C=.5)),
    ('KNN', KNeighborsRegressor(n_neighbors=2)),
    ('Multi Layer Perception',
     MLPRegressor(activation='relu', hidden_layer_sizes=(10, 100), alpha=0.001, random_state=42, early_stopping=False)),
    ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('AdaBoost', AdaBoostRegressor(n_estimators=100, random_state=42)),
    ('Bagging', BaggingRegressor(n_estimators=100, random_state=42)),
    ('Extra Trees', ExtraTreesRegressor(n_estimators=100, random_state=42)),
]
k = 10
# Compare ensemble methods using cross-validation
for name, model in ensemble_methods:
    scores = cross_val_score(model, X, y, cv=k, scoring='neg_mean_squared_error')
    predicted = cross_val_predict(model, X, y, cv=k)
    correlation_coefficient = r2_score(y, predicted)
    mse_scores = -scores  # Convert negative MSE scores to positive
    rmse_scores = np.sqrt(mse_scores)
    r2_scores = cross_val_score(model, X, y, cv=k, scoring='r2')
    # Relative Absolute Error
    absolute_errors = np.abs(y - predicted)
    mean_absolute_error = np.mean(absolute_errors)
    mean_true_values = np.mean(y)
    relative_absolute_error = mean_absolute_error / mean_true_values
    # Calculate the squared errors
    squared_errors = np.square(y - predicted)
    mean_squared_error = np.mean(squared_errors)
    mean_true_values = np.mean(y)
    root_relative_squared_error = np.sqrt(mean_squared_error) / mean_true_values


    print(f"{name} - Correlation Coefficient: {correlation_coefficient:.2f} (+/- {np.std(correlation_coefficient):.2f})")
    print(f"{name} - Mean MSE: {np.mean(mse_scores):.2f} (+/- {np.std(mse_scores):.2f})")
    print(f"{name} - Mean Absolute Error: {np.mean(mse_scores):.2f} (+/- {np.std(mse_scores):.2f})")
    print(f"{name} - Root Mean Squared Error: {np.mean(rmse_scores):.2f} (+/- {np.std(rmse_scores):.2f})")
    print(f"{name} - Mean R-squared (R2): {np.mean(r2_scores):.2f} (+/- {np.std(r2_scores):.2f})")
    print(f"{name} - Relative Absolute Error: {relative_absolute_error:.2f} (+/- {np.std(relative_absolute_error):.2f})")
    print(f"{name} - Root Relative Squared Error (RRSE): {root_relative_squared_error:.2f} (+/- {np.std(relative_absolute_error):.2f})")

    opt = pd.concat([opt, pd.DataFrame.from_records([{'Model': f"{name}",
                                                      'Correlation Coefficient': correlation_coefficient,
                                                      'MAE': np.mean(mse_scores),
                                                      'RMSE': np.mean(rmse_scores),
                                                      'RAE': relative_absolute_error,
                                                      'RRSE': root_relative_squared_error}])])




# opt.to_csv('All_ML_models_Results_v3_Poly_Depth(6features).csv', mode='w', index=False)
opt.to_csv('All_ML_models_Results_v3_Poly_Depth(12features).csv', mode='w',index=False)