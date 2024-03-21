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

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import cross_val_score, KFold
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, ExpSineSquared, RationalQuadratic
from sklearn.model_selection import GridSearchCV, KFold

# df = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_polyExp(12features)Normalized.csv')

output='Depth'
# input_file='DataAug/LPBF_Final_Dataset/LPBF_Dataset_Normalized_Depth.csv'

# output = 'Width'
input_file = f'D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Normalized_{output}.csv'

df = pd.read_csv(input_file)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

degrees = [
    C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)),
    C(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-2, 1e2), nu=0.5),
    C(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-2, 1e2), nu=1.5),
    C(1.0, (1e-3, 1e3)) * ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e2)),
    C(1.0, (1e-3, 1e3)) * RationalQuadratic(1.0, 1.0, alpha_bounds=(1e-2, 1e2)),
    DotProduct() + WhiteKernel(),
    RationalQuadratic(),
    RBF()
]

# Perform 10-fold cross-validation for kNN
# degrees = [2,3,4,5,10,15,20,25,30,35,40,45,50]  # You can experiment with different k
cv = KFold(n_splits=10, shuffle=True, random_state=42)

poly_list = []

for degree in degrees:
    model = GaussianProcessRegressor(kernel=degree)

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
        'Kernel': degree,
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
filename = f'LPBF_HP_GaussianProcess_results_{output}_{timestamp}.csv'

poly_results.to_csv(filename, index=False)