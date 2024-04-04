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
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, ExpSineSquared, RationalQuadratic
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
from sklearn.model_selection import cross_val_score, KFold
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# !###############################################!#
# !                                               !#
# !####################  Width      ##############!#
# !                   Base Models                  !#
# !###############################################!#


# Load the dataset as an example
output = 'Width'
input_file = f'D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Normalized_{output}.csv'
df = pd.read_csv(input_file)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# Function to create a polynomial regression model
def polynomial_regression(degree):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression())


# Perform 10-fold cross-validation
cv = KFold(n_splits=10, shuffle=True, random_state=42)


c1 = ["Width"]
c2 = ["------"]
c3 = ["Width"]
c4 = ["Results"]
c5 = ["------"]
c6 = ["------"]

opt_data = {'Model': c1,
            'Correlation Coefficient': c2,
            'MAE': c3,
            'RMSE': c4,
            'RAE': c5,
            'RRSE':c6 }
opt = pd.DataFrame(opt_data)




# Create a list of ensemble regressors
ensemble_methods = [
    ('Gaussian Process', GaussianProcessRegressor()),
    ('Linear Regression', LinearRegression()),
    ('Polynomial Regression', polynomial_regression(1)),
    ('Support Vector Regression', SVR()),
    ('KNN', KNeighborsRegressor()),
    ('Multi Layer Perception', MLPRegressor()),
    ('Random Forest', RandomForestRegressor()),
    ('Gradient Boosting', GradientBoostingRegressor()),
    ('AdaBoost', AdaBoostRegressor()),
    ('Bagging', BaggingRegressor()),
    ('Extra Trees', ExtraTreesRegressor()),
]

# Compare ensemble methods using cross-validation
for name, model in ensemble_methods:

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

    print("Width Base")
    # print(f"{name} - Correlation Coefficient: {corr}")
    print(f"{name} - Mean Absolute Error: {mae}")
    # print(f"{name} - Root Mean Squared Error: {rmse}")
    # print(f"{name} - Relative Absolute Error: {rae}")
    # print(f"{name} - Root Relative Squared Error (RRSE): {rrse}")

    opt = pd.concat([opt, pd.DataFrame.from_records([{'Model': f"{name}",
                                                      'Correlation Coefficient': corr,
                                                      'MAE': mae,
                                                      'RMSE': rmse,
                                                      'RAE': rae,
                                                      'RRSE': rrse
                                                      }])])


# !####################  Depth      ##############!#
# !                   Base Models                 !#
# !###############################################!#
output = 'Depth'
input_file = f'D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Normalized_{output}.csv'
df = pd.read_csv(input_file)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

opt = pd.concat([opt, pd.DataFrame.from_records([{'Model': "Depth",
                                                  'Correlation Coefficient': "------",
                                                  'MAE': "Depth",
                                                  'RMSE': "Results",
                                                  'RAE': "------",
                                                  'RRSE': "------"}])])


# Create a list of ensemble regressors
ensemble_methods = [
    ('Gaussian Process', GaussianProcessRegressor()),
    ('Linear Regression', LinearRegression()),
    ('Polynomial Regression', polynomial_regression(1)),
    ('Support Vector Regression', SVR()),
    ('KNN', KNeighborsRegressor()),
    ('Multi Layer Perception', MLPRegressor()),
    ('Random Forest', RandomForestRegressor()),
    ('Gradient Boosting', GradientBoostingRegressor()),
    ('AdaBoost', AdaBoostRegressor()),
    ('Bagging', BaggingRegressor()),
    ('Extra Trees', ExtraTreesRegressor()),
]

# Perform 10-fold cross-validation
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Compare ensemble methods using cross-validation
for name, model in ensemble_methods:

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

    print("Depth Base")
    # print(f"{name} - Correlation Coefficient: {corr}")
    print(f"{name} - Mean Absolute Error: {mae}")
    # print(f"{name} - Root Mean Squared Error: {rmse}")
    # print(f"{name} - Relative Absolute Error: {rae}")
    # print(f"{name} - Root Relative Squared Error (RRSE): {rrse}")

    opt = pd.concat([opt, pd.DataFrame.from_records([{'Model': f"{name}",
                                                      'Correlation Coefficient': corr,
                                                      'MAE': mae,
                                                      'RMSE': rmse,
                                                      'RAE': rae,
                                                      'RRSE': rrse
                                                      }])])

opt.to_csv('LPBF_ML_Model_Evaluation_BaseModel.csv',mode='w', index=False)



# !###############################################!#
# !                                               !#
# !####################  Width      ##############!#
# !                   HyperParameter Tuning        !#
# !###############################################!#


# Load the dataset as an example
output = 'Width'
input_file = f'D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Normalized_{output}.csv'
df = pd.read_csv(input_file)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Function to create a polynomial regression model
def polynomial_regression(degree):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression())


# Perform 10-fold cross-validation
cv = KFold(n_splits=10, shuffle=True, random_state=42)


c1 = ["Width"]
c2 = ["------"]
c3 = ["Width"]
c4 = ["Results"]
c5 = ["------"]
c6 = ["------"]


opt_data = {'Model': c1,
            'Correlation Coefficient': c2,
            'MAE': c3,
            'RMSE': c4,
            'RAE': c5,
            'RRSE':c6 }
opt = pd.DataFrame(opt_data)

activation_values = ['relu'] #default relu
solver_values = ['lbfgs'] #default adam
learning_rate_values = ['constant'] #default constant
# Create a list of ensemble regressors
ensemble_methods = [
    ('Gaussian Process', GaussianProcessRegressor(C(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-2, 1e2), nu=0.5))),
    ('Linear Regression', LinearRegression()),
    ('Polynomial Regression', polynomial_regression(2)),
    ('Support Vector Regression', make_pipeline(StandardScaler(), SVR(kernel='rbf', C=10, gamma=0.01))),
    ('KNN', KNeighborsRegressor(n_neighbors=2)),
    ('Multi Layer Perception', MLPRegressor(hidden_layer_sizes=50,
                                            activation='relu',
                                            solver='lbfgs',
                                            alpha=0.01,
                                            batch_size=2,
                                            learning_rate='constant',
                                            max_iter=600,
                                            early_stopping=False,
                                                      )),

    ('Random Forest', RandomForestRegressor(n_estimators=200, max_depth=None,
                                            min_samples_split=2, min_samples_leaf=1,
                                              )),
    ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, max_depth=10,
                                                   min_samples_split=2, min_samples_leaf=4,
                                                   )),
    ('AdaBoost', AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=5),
                                   n_estimators=30, learning_rate=0.01)),
    ('Bagging', BaggingRegressor(estimator=None, n_estimators=200,
                                 max_samples=0.5,max_features=4)),
    ('Extra Trees', ExtraTreesRegressor(n_estimators= 10, max_depth= None,
                                         min_samples_split= 2, min_samples_leaf= 1)),
]

cv = KFold(n_splits=10, shuffle=True, random_state=42)
# Compare ensemble methods using cross-validation
for name, model in ensemble_methods:

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

    print("Width HP")
    # print(f"{name} - Correlation Coefficient: {corr}")
    print(f"{name} - Mean Absolute Error: {mae}")
    # print(f"{name} - Root Mean Squared Error: {rmse}")
    # print(f"{name} - Relative Absolute Error: {rae}")
    # print(f"{name} - Root Relative Squared Error (RRSE): {rrse}")

    opt = pd.concat([opt, pd.DataFrame.from_records([{'Model': f"{name}",
                                                      'Correlation Coefficient': corr,
                                                      'MAE': mae,
                                                      'RMSE': rmse,
                                                      'RAE': rae,
                                                      'RRSE': rrse
                                                      }])])


# !####################  Depth      ##############!#
# !                   HP Tuning                   !#
# !###############################################!#
output = 'Depth'
input_file = f'D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Normalized_{output}.csv'
df = pd.read_csv(input_file)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

opt = pd.concat([opt, pd.DataFrame.from_records([{'Model': "Depth",
                                                  'Correlation Coefficient': "------",
                                                  'MAE': "Depth",
                                                  'RMSE': "Results",
                                                  'RAE': "------",
                                                  'RRSE': "------"}])])


# Create a list of ensemble regressors
ensemble_methods = [
    ('Gaussian Process', GaussianProcessRegressor(C(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-2, 1e2), nu=0.5))),
    ('Linear Regression', LinearRegression()),
    ('Polynomial Regression', polynomial_regression(2)),
    ('Support Vector Regression', make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma=0.0001))),
    ('KNN', KNeighborsRegressor(n_neighbors=2)),
    ('Multi Layer Perception', MLPRegressor(hidden_layer_sizes=50,
                                              activation='relu',
                                              solver='lbfgs',
                                              alpha=0.01,
                                              batch_size=2,
                                              learning_rate='constant',
                                              max_iter=600,
                                              early_stopping=False,
                                                      )),

    ('Random Forest', RandomForestRegressor(n_estimators=200, max_depth=20,
                                            min_samples_split=2, min_samples_leaf=1,
                                              )),
    ('Gradient Boosting', GradientBoostingRegressor(n_estimators=50, max_depth=20,
                                              min_samples_split=10, min_samples_leaf=4,
                                              )),
    ('AdaBoost', AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=5),
                                   n_estimators=100, learning_rate=0.1)),
    ('Bagging', BaggingRegressor(estimator=None, n_estimators=200,
                                 max_samples=0.5, max_features=4)),
    ('Extra Trees', ExtraTreesRegressor(n_estimators= 100, max_depth= None,
                                        min_samples_split= 2, min_samples_leaf= 1,
                                        )),
]
cv = KFold(n_splits=10, shuffle=True, random_state=42)
# Compare ensemble methods using cross-validation
for name, model in ensemble_methods:

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

    print("Depth HP")
    # print(f"{name} - Correlation Coefficient: {corr}")
    print(f"{name} - Mean Absolute Error: {mae}")
    # print(f"{name} - Root Mean Squared Error: {rmse}")
    # print(f"{name} - Relative Absolute Error: {rae}")
    # print(f"{name} - Root Relative Squared Error (RRSE): {rrse}")

    opt = pd.concat([opt, pd.DataFrame.from_records([{'Model': f"{name}",
                                                      'Correlation Coefficient': corr,
                                                      'MAE': mae,
                                                      'RMSE': rmse,
                                                      'RAE': rae,
                                                      'RRSE': rrse
                                                      }])])

opt.to_csv('LPBF_ML_Model_Evaluation_HpTuning.csv',mode='w', index=False)





#
#
# # !###############################################!#
# # !                                               !#
# # !####################  Width      ##############!#
# # !                   Poly 6 Features             !#
# # !###############################################!#
#
#
# # Load the dataset as an example
# df = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_polyExp(6features)Normalized.csv')
# X = df[['Laser Power [W]', 'Scanning Speed [mm/s]', 'Layer Thickness [um]', 'Spot Size [um]', 'Porosity [%]','LPxLT','LPxSS','LPxP','SSpxLT','SSpxSS','SSpxP']].values
# y = df['Max Melt Pool Width [um]'].values
# y1 = df['Max Melt Pool Depth [um]'].values
#
#
# poly = PolynomialFeatures(degree=7, include_bias=False)
# poly_features = poly.fit_transform(X)
# poly_reg_model = LinearRegression()
#
#
# c1 = ["Width"]
# c2 = ["Width"]
# c3 = ["Results"]
# c4 = ["Poly"]
# c5 = ["6 features"]
# c6 = ["------"]
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
#     ('Gaussian Process', GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(),random_state=42).fit(X, y)),
#     ('Linear Regression', LinearRegression()),
#     ('Polynomial Regression', poly_reg_model.fit(poly_features, y)),
#     ('Support Vector Regression', SVR(kernel='linear', C=.5)),
#     ('KNN', KNeighborsRegressor(n_neighbors=2)),
#     ('Multi Layer Perception', MLPRegressor(activation='relu',hidden_layer_sizes=(10, 100),alpha=0.001,random_state=42,early_stopping=False)),
#     ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42)),
#     ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
#     ('AdaBoost', AdaBoostRegressor(n_estimators=100, random_state=42)),
#     ('Bagging', BaggingRegressor(n_estimators=100, random_state=42)),
#     ('Extra Trees', ExtraTreesRegressor(n_estimators=100, random_state=42)),
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
# # !####################  Depth      ##############!#
# # !                   poly 6 features             !#
# # !###############################################!#
#
# opt = pd.concat([opt, pd.DataFrame.from_records([{'Model': "Depth",
#                                                   'Correlation Coefficient': "Depth",
#                                                   'MAE': "Results",
#                                                   'RMSE': "Poly",
#                                                   'RAE': "6 features",
#                                                   'RRSE': "------"}])])
#
# # Create a list of ensemble regressors
# ensemble_methods = [
#     ('Gaussian Process', GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(),random_state=42).fit(X, y)),
#     ('Linear Regression', LinearRegression()),
#     ('Polynomial Regression', poly_reg_model.fit(poly_features, y1)),
#     ('Support Vector Regression', SVR(kernel='linear', C=.5)),
#     ('KNN', KNeighborsRegressor(n_neighbors=2)),
#     ('Multi Layer Perception', MLPRegressor(activation='relu',hidden_layer_sizes=(10, 100),alpha=0.001,random_state=42,early_stopping=False)),
#     ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42)),
#     ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
#     ('AdaBoost', AdaBoostRegressor(n_estimators=100, random_state=42)),
#     ('Bagging', BaggingRegressor(n_estimators=100, random_state=42)),
#     ('Extra Trees', ExtraTreesRegressor(n_estimators=100, random_state=42)),
# ]
# k = 10
# # Compare ensemble methods using cross-validation
# for name, model in ensemble_methods:
#     scores = cross_val_score(model, X, y1, cv=k, scoring='neg_mean_squared_error')
#     predicted = cross_val_predict(model, X, y1, cv=k)
#     correlation_coefficient = r2_score(y1, predicted)
#     mse_scores = -scores  # Convert negative MSE scores to positive
#     rmse_scores = np.sqrt(mse_scores)
#     r2_scores = cross_val_score(model, X, y1, cv=k, scoring='r2')
#     # Relative Absolute Error
#     absolute_errors = np.abs(y1 - predicted)
#     mean_absolute_error = np.mean(absolute_errors)
#     mean_true_values = np.mean(y1)
#     relative_absolute_error = mean_absolute_error / mean_true_values
#     # Calculate the squared errors
#     squared_errors = np.square(y1 - predicted)
#     mean_squared_error = np.mean(squared_errors)
#     mean_true_values = np.mean(y1)
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
# opt.to_csv('LPBF_ML_Model_Evaluation_poly6features.csv',mode='w', index=False)
#
#
# # !###############################################!#
# # !                                               !#
# # !####################  Width      ##############!#
# # !                   Poly 12 Features             !#
# # !###############################################!#
#
#
# # Load the dataset as an example
# df = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_polyExp(12features)Normalized.csv')
# X = df[['Laser Power [W]', 'Scanning Speed [mm/s]', 'Layer Thickness [um]', 'Spot Size [um]', 'Porosity [%]','LPxLT','LPxSS','LPxP','SSpxLT','SSpxSS','SSpxP','LP2xLT','LP2xSS','LP2xP','SSp2xLT','SSp2xSS','SSp2xP']].values
# y = df['Max Melt Pool Width [um]'].values
# y1 = df['Max Melt Pool Depth [um]'].values
#
#
# poly = PolynomialFeatures(degree=7, include_bias=False)
# poly_features = poly.fit_transform(X)
# poly_reg_model = LinearRegression()
#
#
# c1 = ["Width"]
# c2 = ["Width"]
# c3 = ["Results"]
# c4 = ["Poly"]
# c5 = ["6 features"]
# c6 = ["------"]
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
#     ('Gaussian Process', GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(),random_state=42).fit(X, y)),
#     ('Linear Regression', LinearRegression()),
#     ('Polynomial Regression', poly_reg_model.fit(poly_features, y)),
#     ('Support Vector Regression', SVR(kernel='linear', C=.5)),
#     ('KNN', KNeighborsRegressor(n_neighbors=2)),
#     ('Multi Layer Perception', MLPRegressor(activation='relu',hidden_layer_sizes=(10, 100),alpha=0.001,random_state=42,early_stopping=False)),
#     ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42)),
#     ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
#     ('AdaBoost', AdaBoostRegressor(n_estimators=100, random_state=42)),
#     ('Bagging', BaggingRegressor(n_estimators=100, random_state=42)),
#     ('Extra Trees', ExtraTreesRegressor(n_estimators=100, random_state=42)),
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
# # !####################  Depth      ##############!#
# # !                   poly 12 features             !#
# # !###############################################!#
#
# opt = pd.concat([opt, pd.DataFrame.from_records([{'Model': "Depth",
#                                                   'Correlation Coefficient': "Depth",
#                                                   'MAE': "Results",
#                                                   'RMSE': "Poly",
#                                                   'RAE': "6 features",
#                                                   'RRSE': "------"}])])
#
# # Create a list of ensemble regressors
# ensemble_methods = [
#     ('Gaussian Process', GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(),random_state=42).fit(X, y)),
#     ('Linear Regression', LinearRegression()),
#     ('Polynomial Regression', poly_reg_model.fit(poly_features, y1)),
#     ('Support Vector Regression', SVR(kernel='linear', C=.5)),
#     ('KNN', KNeighborsRegressor(n_neighbors=2)),
#     ('Multi Layer Perception', MLPRegressor(activation='relu',hidden_layer_sizes=(10, 100),alpha=0.001,random_state=42,early_stopping=False)),
#     ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42)),
#     ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
#     ('AdaBoost', AdaBoostRegressor(n_estimators=100, random_state=42)),
#     ('Bagging', BaggingRegressor(n_estimators=100, random_state=42)),
#     ('Extra Trees', ExtraTreesRegressor(n_estimators=100, random_state=42)),
# ]
# k = 10
# # Compare ensemble methods using cross-validation
# for name, model in ensemble_methods:
#     scores = cross_val_score(model, X, y1, cv=k, scoring='neg_mean_squared_error')
#     predicted = cross_val_predict(model, X, y1, cv=k)
#     correlation_coefficient = r2_score(y1, predicted)
#     mse_scores = -scores  # Convert negative MSE scores to positive
#     rmse_scores = np.sqrt(mse_scores)
#     r2_scores = cross_val_score(model, X, y1, cv=k, scoring='r2')
#     # Relative Absolute Error
#     absolute_errors = np.abs(y1 - predicted)
#     mean_absolute_error = np.mean(absolute_errors)
#     mean_true_values = np.mean(y1)
#     relative_absolute_error = mean_absolute_error / mean_true_values
#     # Calculate the squared errors
#     squared_errors = np.square(y1 - predicted)
#     mean_squared_error = np.mean(squared_errors)
#     mean_true_values = np.mean(y1)
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
# opt.to_csv('LPBF_ML_Model_Evaluation_poly12features.csv',mode='w', index=False)
#
#
# # !###############################################!#
# # !                                               !#
# # !####################  Width      ##############!#
# # !                   Synthetic 1000              !#
# # !###############################################!#
#
#
# # Load the dataset as an example
# df = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Synthetic_1000_Normalized.csv')
# X = df[['Laser Power [W]', 'Scanning Speed [mm/s]', 'Layer Thickness [um]', 'Spot Size [um]', 'Porosity [%]']].values
# y = df['Max Melt Pool Width [um]'].values
# y1 = df['Max Melt Pool Depth [um]'].values
#
#
# poly = PolynomialFeatures(degree=7, include_bias=False)
# poly_features = poly.fit_transform(X)
# poly_reg_model = LinearRegression()
#
#
# c1 = ["Width"]
# c2 = ["Width"]
# c3 = ["Results"]
# c4 = ["Synthetic"]
# c5 = ["1000"]
# c6 = ["------"]
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
#     ('Gaussian Process', GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(),random_state=42).fit(X, y)),
#     ('Linear Regression', LinearRegression()),
#     ('Polynomial Regression', poly_reg_model.fit(poly_features, y)),
#     ('Support Vector Regression', SVR(kernel='linear', C=.5)),
#     ('KNN', KNeighborsRegressor(n_neighbors=2)),
#     ('Multi Layer Perception', MLPRegressor(activation='relu',hidden_layer_sizes=(10, 100),alpha=0.001,random_state=42,early_stopping=False)),
#     ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42)),
#     ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
#     ('AdaBoost', AdaBoostRegressor(n_estimators=100, random_state=42)),
#     ('Bagging', BaggingRegressor(n_estimators=100, random_state=42)),
#     ('Extra Trees', ExtraTreesRegressor(n_estimators=100, random_state=42)),
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
# # !####################  Depth      ##############!#
# # !                   Synthetic 1000              !#
# # !###############################################!#
#
# opt = pd.concat([opt, pd.DataFrame.from_records([{'Model': "Depth",
#                                                   'Correlation Coefficient': "Depth",
#                                                   'MAE': "Results",
#                                                   'RMSE': "Synthetic",
#                                                   'RAE': "1000",
#                                                   'RRSE': "------"}])])
#
# # Create a list of ensemble regressors
# ensemble_methods = [
#     ('Gaussian Process', GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(),random_state=42).fit(X, y)),
#     ('Linear Regression', LinearRegression()),
#     ('Polynomial Regression', poly_reg_model.fit(poly_features, y1)),
#     ('Support Vector Regression', SVR(kernel='linear', C=.5)),
#     ('KNN', KNeighborsRegressor(n_neighbors=2)),
#     ('Multi Layer Perception', MLPRegressor(activation='relu',hidden_layer_sizes=(10, 100),alpha=0.001,random_state=42,early_stopping=False)),
#     ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42)),
#     ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
#     ('AdaBoost', AdaBoostRegressor(n_estimators=100, random_state=42)),
#     ('Bagging', BaggingRegressor(n_estimators=100, random_state=42)),
#     ('Extra Trees', ExtraTreesRegressor(n_estimators=100, random_state=42)),
# ]
# k = 10
# # Compare ensemble methods using cross-validation
# for name, model in ensemble_methods:
#     scores = cross_val_score(model, X, y1, cv=k, scoring='neg_mean_squared_error')
#     predicted = cross_val_predict(model, X, y1, cv=k)
#     correlation_coefficient = r2_score(y1, predicted)
#     mse_scores = -scores  # Convert negative MSE scores to positive
#     rmse_scores = np.sqrt(mse_scores)
#     r2_scores = cross_val_score(model, X, y1, cv=k, scoring='r2')
#     # Relative Absolute Error
#     absolute_errors = np.abs(y1 - predicted)
#     mean_absolute_error = np.mean(absolute_errors)
#     mean_true_values = np.mean(y1)
#     relative_absolute_error = mean_absolute_error / mean_true_values
#     # Calculate the squared errors
#     squared_errors = np.square(y1 - predicted)
#     mean_squared_error = np.mean(squared_errors)
#     mean_true_values = np.mean(y1)
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
# opt.to_csv('LPBF_ML_Model_Evaluation_Synthetic_1000.csv',mode='w', index=False)
#
#
# # !###############################################!#
# # !                                               !#
# # !####################  Width      ##############!#
# # !                   Synthetic 2000              !#
# # !###############################################!#
#
#
# # Load the dataset as an example
# df = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Synthetic_2000_Normalized.csv')
# X = df[['Laser Power [W]', 'Scanning Speed [mm/s]', 'Layer Thickness [um]', 'Spot Size [um]', 'Porosity [%]']].values
# y = df['Max Melt Pool Width [um]'].values
# y1 = df['Max Melt Pool Depth [um]'].values
#
#
# poly = PolynomialFeatures(degree=7, include_bias=False)
# poly_features = poly.fit_transform(X)
# poly_reg_model = LinearRegression()
#
#
# c1 = ["Width"]
# c2 = ["Width"]
# c3 = ["Results"]
# c4 = ["Synthetic"]
# c5 = ["2000"]
# c6 = ["------"]
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
#     ('Gaussian Process', GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(),random_state=42).fit(X, y)),
#     ('Linear Regression', LinearRegression()),
#     ('Polynomial Regression', poly_reg_model.fit(poly_features, y)),
#     ('Support Vector Regression', SVR(kernel='linear', C=.5)),
#     ('KNN', KNeighborsRegressor(n_neighbors=2)),
#     ('Multi Layer Perception', MLPRegressor(activation='relu',hidden_layer_sizes=(10, 100),alpha=0.001,random_state=42,early_stopping=False)),
#     ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42)),
#     ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
#     ('AdaBoost', AdaBoostRegressor(n_estimators=100, random_state=42)),
#     ('Bagging', BaggingRegressor(n_estimators=100, random_state=42)),
#     ('Extra Trees', ExtraTreesRegressor(n_estimators=100, random_state=42)),
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
# # !####################  Depth      ##############!#
# # !                   Synthetic 2000              !#
# # !###############################################!#
#
# opt = pd.concat([opt, pd.DataFrame.from_records([{'Model': "Depth",
#                                                   'Correlation Coefficient': "Depth",
#                                                   'MAE': "Results",
#                                                   'RMSE': "Synthetic",
#                                                   'RAE': "2000",
#                                                   'RRSE': "------"}])])
#
# # Create a list of ensemble regressors
# ensemble_methods = [
#     ('Gaussian Process', GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(),random_state=42).fit(X, y)),
#     ('Linear Regression', LinearRegression()),
#     ('Polynomial Regression', poly_reg_model.fit(poly_features, y1)),
#     ('Support Vector Regression', SVR(kernel='linear', C=.5)),
#     ('KNN', KNeighborsRegressor(n_neighbors=2)),
#     ('Multi Layer Perception', MLPRegressor(activation='relu',hidden_layer_sizes=(10, 100),alpha=0.001,random_state=42,early_stopping=False)),
#     ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42)),
#     ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
#     ('AdaBoost', AdaBoostRegressor(n_estimators=100, random_state=42)),
#     ('Bagging', BaggingRegressor(n_estimators=100, random_state=42)),
#     ('Extra Trees', ExtraTreesRegressor(n_estimators=100, random_state=42)),
# ]
# k = 10
# # Compare ensemble methods using cross-validation
# for name, model in ensemble_methods:
#     scores = cross_val_score(model, X, y1, cv=k, scoring='neg_mean_squared_error')
#     predicted = cross_val_predict(model, X, y1, cv=k)
#     correlation_coefficient = r2_score(y1, predicted)
#     mse_scores = -scores  # Convert negative MSE scores to positive
#     rmse_scores = np.sqrt(mse_scores)
#     r2_scores = cross_val_score(model, X, y1, cv=k, scoring='r2')
#     # Relative Absolute Error
#     absolute_errors = np.abs(y1 - predicted)
#     mean_absolute_error = np.mean(absolute_errors)
#     mean_true_values = np.mean(y1)
#     relative_absolute_error = mean_absolute_error / mean_true_values
#     # Calculate the squared errors
#     squared_errors = np.square(y1 - predicted)
#     mean_squared_error = np.mean(squared_errors)
#     mean_true_values = np.mean(y1)
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
# opt.to_csv('LPBF_ML_Model_Evaluation_Synthetic_2000.csv',mode='w', index=False)
#
#
# # !###############################################!#
# # !                                               !#
# # !####################  Width      ##############!#
# # !                   Synthetic 3000              !#
# # !###############################################!#
#
#
# # Load the dataset as an example
# df = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Synthetic_3000_Normalized.csv')
# X = df[['Laser Power [W]', 'Scanning Speed [mm/s]', 'Layer Thickness [um]', 'Spot Size [um]', 'Porosity [%]']].values
# y = df['Max Melt Pool Width [um]'].values
# y1 = df['Max Melt Pool Depth [um]'].values
#
#
# poly = PolynomialFeatures(degree=7, include_bias=False)
# poly_features = poly.fit_transform(X)
# poly_reg_model = LinearRegression()
#
#
# c1 = ["Width"]
# c2 = ["Width"]
# c3 = ["Results"]
# c4 = ["Synthetic"]
# c5 = ["3000"]
# c6 = ["------"]
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
#     ('Gaussian Process', GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(),random_state=42).fit(X, y)),
#     ('Linear Regression', LinearRegression()),
#     ('Polynomial Regression', poly_reg_model.fit(poly_features, y)),
#     ('Support Vector Regression', SVR(kernel='linear', C=.5)),
#     ('KNN', KNeighborsRegressor(n_neighbors=2)),
#     ('Multi Layer Perception', MLPRegressor(activation='relu',hidden_layer_sizes=(10, 100),alpha=0.001,random_state=42,early_stopping=False)),
#     ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42)),
#     ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
#     ('AdaBoost', AdaBoostRegressor(n_estimators=100, random_state=42)),
#     ('Bagging', BaggingRegressor(n_estimators=100, random_state=42)),
#     ('Extra Trees', ExtraTreesRegressor(n_estimators=100, random_state=42)),
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
# # !####################  Depth      ##############!#
# # !                   Synthetic 3000              !#
# # !###############################################!#
#
# opt = pd.concat([opt, pd.DataFrame.from_records([{'Model': "Depth",
#                                                   'Correlation Coefficient': "Depth",
#                                                   'MAE': "Results",
#                                                   'RMSE': "Synthetic",
#                                                   'RAE': "3000",
#                                                   'RRSE': "------"}])])
#
# # Create a list of ensemble regressors
# ensemble_methods = [
#     ('Gaussian Process', GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(),random_state=42).fit(X, y)),
#     ('Linear Regression', LinearRegression()),
#     ('Polynomial Regression', poly_reg_model.fit(poly_features, y1)),
#     ('Support Vector Regression', SVR(kernel='linear', C=.5)),
#     ('KNN', KNeighborsRegressor(n_neighbors=2)),
#     ('Multi Layer Perception', MLPRegressor(activation='relu',hidden_layer_sizes=(10, 100),alpha=0.001,random_state=42,early_stopping=False)),
#     ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42)),
#     ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
#     ('AdaBoost', AdaBoostRegressor(n_estimators=100, random_state=42)),
#     ('Bagging', BaggingRegressor(n_estimators=100, random_state=42)),
#     ('Extra Trees', ExtraTreesRegressor(n_estimators=100, random_state=42)),
# ]
# k = 10
# # Compare ensemble methods using cross-validation
# for name, model in ensemble_methods:
#     scores = cross_val_score(model, X, y1, cv=k, scoring='neg_mean_squared_error')
#     predicted = cross_val_predict(model, X, y1, cv=k)
#     correlation_coefficient = r2_score(y1, predicted)
#     mse_scores = -scores  # Convert negative MSE scores to positive
#     rmse_scores = np.sqrt(mse_scores)
#     r2_scores = cross_val_score(model, X, y1, cv=k, scoring='r2')
#     # Relative Absolute Error
#     absolute_errors = np.abs(y1 - predicted)
#     mean_absolute_error = np.mean(absolute_errors)
#     mean_true_values = np.mean(y1)
#     relative_absolute_error = mean_absolute_error / mean_true_values
#     # Calculate the squared errors
#     squared_errors = np.square(y1 - predicted)
#     mean_squared_error = np.mean(squared_errors)
#     mean_true_values = np.mean(y1)
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
# opt.to_csv('LPBF_ML_Model_Evaluation_Synthetic_3000.csv',mode='w', index=False)