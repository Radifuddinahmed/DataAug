# Import necessary libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_predict, KFold
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
from sklearn.tree import DecisionTreeRegressor
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

import timeit
start = timeit.default_timer()

#!###############################################!#
#!                                               !#
#!####################  Depth ###################!#
#!                                               !#
#!###############################################!#

# Load the dataset as an example
# df = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Original_Depth.csv')
df = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Original - Copy.csv')
X1 = df[['Laser Power [W]', 'Scanning Speed [mm/s]', 'Layer Thickness [um]', 'Spot Size [um]', 'Porosity [%]']].values
y1 = df['Max Melt Pool Depth [um]'].values

from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

# Create a Gaussian Process Regressor
GP_regressor = GaussianProcessRegressor(C(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-2, 1e2), nu=0.5))
# # Create a Gaussian Process Regressor
# GP_regressor = BaggingRegressor(
#     estimator=None,
#     n_estimators=20,
#     max_samples=0.7,
#     max_features=1.0,
#     bootstrap=True,
#     bootstrap_features=False,
#     oob_score=False,
#     warm_start=False,
#     n_jobs=None,
#     random_state=42,
#     verbose=0,
#     base_estimator="deprecated",
# )

# GP_regressor.fit(X1_train, y1_train)
#
# # Perform K-fold cross-validated predictions
# y1_pred_cv = GP_regressor.predict(X1_test)
#
#
# opt11 = pd.DataFrame({'actual': y1_test})
# opt21= pd.DataFrame({'pred':y1_pred_cv})
# opt31= pd.concat([opt11,opt21], axis=1)
#
# # append data frame to CSV file
# opt31.to_csv('LPBF_ML_Model_Prediction_Depth_GP.csv', mode='w', index=False, header=False)
#
# # print message
# print("Data appended successfully.")

###########################
# code for validation table   Depth
###########################

# Load the dataset as an example - depth - et
# df = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Original_Depth.csv')
df = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Original - Copy.csv')
X2 = df[['Laser Power [W]', 'Scanning Speed [mm/s]', 'Layer Thickness [um]', 'Spot Size [um]', 'Porosity [%]']].values
y2 = df['Max Melt Pool Depth [um]'].values

# Create a Gaussian Process Regressor
GP_regressor = GaussianProcessRegressor(C(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-2, 1e2), nu=0.5))
# Create a Gaussian Process Regressor
# GP_regressor = GaussianProcessRegressor()


GP_regressor.fit(X2,y2)

val_data = [[200,300,70,58,50]]
prediction_GP_depth = GP_regressor.predict(val_data)

# print message
print("depth GP: ")
## in dataset 189.5
print(prediction_GP_depth)

stop = timeit.default_timer()

print('Time: ', stop - start)
