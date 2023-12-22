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


#!###############################################!#
#!                                               !#
#!####################  Width ###################!#
#!                                               !#
#!###############################################!#

# Load the dataset as an example
df = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Original_Width.csv')
X = df[['Laser Power [W]', 'Scanning Speed [mm/s]', 'Layer Thickness [um]', 'Spot Size [um]', 'Porosity [%]']].values
y = df['Max Melt Pool Width [um]'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an Extra Trees Regressor
ET_regressor = ExtraTreesRegressor(
                                n_estimators=10, max_depth=None,
                                min_samples_split=2, min_samples_leaf=1,
                                random_state=42,
)

ET_regressor.fit(X_train, y_train)

y_pred_cv = ET_regressor.predict(X_test)


opt1 = pd.DataFrame({'actual':y_test})
opt2= pd.DataFrame({'pred':y_pred_cv})
opt3= pd.concat([opt1,opt2], axis=1)

# append data frame to CSV file
opt3.to_csv('LPBF_ML_Model_Prediction_Width_ET.csv', mode='w', index=False, header=False)

# print message
print("Data appended successfully.")


###########################
# code for validation table   Width
###########################
output = 'Width'
input_file = f'D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Original_{output}.csv'
df = pd.read_csv(input_file)
X1 = df.iloc[:, :-1]
y1 = df.iloc[:, -1]


# Create an Extra Trees Regressor
ET_regressor = ExtraTreesRegressor(max_depth=10,min_samples_split=5,n_estimators=10, random_state=42)

# # Create an Extra Trees Regressor
# ET_regressor = ExtraTreesRegressor(
#                                 n_estimators=10, max_depth=None,
#                                 min_samples_split=2, min_samples_leaf=1,
#                                 random_state=42,
# )

ET_regressor.fit(X1, y1)

val_data = [[200,300,70,58,50]]
prediction_ET_width = ET_regressor.predict(val_data)

# print message
print("width ET: ")
print(prediction_ET_width)



