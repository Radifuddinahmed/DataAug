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


# Load the dataset as an example
output = 'Width'
input_file = f'D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Normalized_{output}.csv'
df = pd.read_csv(input_file)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# Function to create a polynomial regression model
def polynomial_regression(degree):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression())


opt_data = []

# Create a list of ensemble regressors
ensemble_methods = [
    ('Gaussian Process', GaussianProcessRegressor(C(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-2, 1e2), nu=0.5),
                                                      random_state=42,
                                                  )),
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
                                            random_state=42,
                                                      )),

    ('Random Forest', RandomForestRegressor(n_estimators=200, max_depth=None,
                                            min_samples_split=2, min_samples_leaf=1,
                                            random_state=42,
                                              )),
    ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, max_depth=10,
                                                   min_samples_split=2, min_samples_leaf=4,
                                                    random_state=42,
                                                   )),
    ('AdaBoost', AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=5),
                                   n_estimators=30, learning_rate=0.01,
                                   random_state=42,
                                   )),
    ('Bagging', BaggingRegressor(
                                estimator=None,
                                n_estimators=200,
                                max_samples=0.9,
                                max_features=1.0,
                                bootstrap=True,
                                bootstrap_features=False,
                                oob_score=False,
                                warm_start=False,
                                n_jobs=None,
                                random_state=42,
                                verbose=0,
                                base_estimator="deprecated",
                                 )),
    # ('Bagging', BaggingRegressor(estimator=None, n_estimators=1,
    #                              max_samples=0.5, max_features=4,
    #                              random_state=42,
    #                              )),
    ('Extra Trees', ExtraTreesRegressor(n_estimators= 10, max_depth= None,
                                         min_samples_split= 2, min_samples_leaf= 1,
                                        random_state=42,
                                        )),



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

    print(f"{output} HP")
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
opt.to_csv(f'LPBF_ML_Model_Evaluation_HpTuning_{output}.csv',mode='w', index=False)

