# Import necessary libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
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
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


# Load the dataset as an example
df = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\LPBF_Dataset_Combined_v2_python_Depth.csv')
X = df[['Laser Power [W]', 'Scanning Speed [mm/s]', 'Layer Thickness [um]', 'Spot Size [um]', 'Porosity [%]']].values
y = df['Max Melt Pool Depth [um]'].values

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
    ('Gaussian Process', GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(),random_state=0).fit(X, y)),
    ('Linear Regression', LinearRegression()),
    ('Polynomial Regression', LinearRegression()),
    ('Support Vector Regression', SVR(kernel='linear', C=.5)),
    ('KNN', KNeighborsRegressor(n_neighbors=2)),
    ('Multi Layer Perception', MLPRegressor(activation='relu',hidden_layer_sizes=(10, 100),alpha=0.001,random_state=20,early_stopping=False)),
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

opt.to_csv('All_ML_models_Results_v2.csv', index=False)