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
output = 'Depth'
input_file = f'D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Normalized_{output}.csv'
df = pd.read_csv(input_file)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# Function to create a polynomial regression model
def polynomial_regression(degree):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression())


# Perform 10-fold cross-validation
cv = KFold(n_splits=10, shuffle=True, random_state=42)



opt_data = []

# Create a list of ensemble regressors
ensemble_methods = [
    ('Gaussian Process', GaussianProcessRegressor(random_state=42)),
    ('Linear Regression', LinearRegression()),
    ('Polynomial Regression', polynomial_regression(1)),
    ('Support Vector Regression', SVR()),
    ('KNN', KNeighborsRegressor()),
    ('Multi Layer Perception', MLPRegressor(random_state=42)),
    ('Random Forest', RandomForestRegressor(random_state=42)),
    ('Gradient Boosting', GradientBoostingRegressor(random_state=42)),
    ('AdaBoost', AdaBoostRegressor(random_state=42)),
    ('Bagging', BaggingRegressor(random_state=42)),
    ('Extra Trees', ExtraTreesRegressor(random_state=42)),
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


    opt_data.append({
        'Model': f"{name}",
        'Correlation Coefficient': corr,
        'MAE': mae,
        'RMSE': rmse,
        'RAE': rae,
        'RRSE': rrse
    })
opt = pd.DataFrame(opt_data)
opt.to_csv(f'LPBF_ML_Model_Evaluation_BaseModel_{output}.csv',mode='w', index=False)

