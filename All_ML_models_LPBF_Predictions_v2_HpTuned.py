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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create an Extra Trees Regressor
ET_regressor = ExtraTreesRegressor(max_depth=10,min_samples_split=5,n_estimators=10, random_state=42)

ET_regressor.fit(X_train, y_train)

y_pred_cv = ET_regressor.predict(X_test)

opt1 = pd.DataFrame({'actual':y_test})
opt2= pd.DataFrame({'pred':y_pred_cv})
opt3= pd.concat([opt1,opt2], axis=1)

# append data frame to CSV file
opt3.to_csv('LPBF_Pred_V2_ET_Width.csv', mode='w', index=False, header=False)

# print message
print("Data appended successfully.")


###########################
# code for validation table   Width
###########################

df = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Original_Width.csv')
X = df[['Laser Power [W]', 'Scanning Speed [mm/s]', 'Layer Thickness [um]', 'Spot Size [um]', 'Porosity [%]']].values
y = df['Max Melt Pool Width [um]'].values

# Create an Extra Trees Regressor
ET_regressor = ExtraTreesRegressor(max_depth=10,min_samples_split=5,n_estimators=10, random_state=42)

ET_regressor.fit(X_train, y_train)

val_data = [[200,300,70,100,50]]
prediction_ET_width = ET_regressor.predict(val_data)

# print message
print("width ET: ")
print(prediction_ET_width)




#!###############################################!#
#!                                               !#
#!####################  Depth ###################!#
#!                                               !#
#!###############################################!#

# Load the dataset as an example
df = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Original_Width.csv')
X1 = df[['Laser Power [W]', 'Scanning Speed [mm/s]', 'Layer Thickness [um]', 'Spot Size [um]', 'Porosity [%]']].values
y1 = df['Max Melt Pool Width [um]'].values

from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=0)


# Create a Random Forest Regressor
bagging_regressor = BaggingRegressor(base_estimator= DecisionTreeRegressor(),n_estimators=200, random_state=42)

bagging_regressor.fit(X1_train, y1_train)

# Perform K-fold cross-validated predictions
y1_pred_cv = bagging_regressor.predict(X1_test)


opt11 = pd.DataFrame({'actual': y1_test})
opt21= pd.DataFrame({'pred':y1_pred_cv})
opt31= pd.concat([opt11,opt21], axis=1)

# append data frame to CSV file
opt31.to_csv('LPBF_Pred_V2_BR_Depth.csv', mode='w', index=False, header=False)

# print message
print("Data appended successfully.")

###########################
# code for validation table   Depth
###########################

# Load the dataset as an example - depth - et
df = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Original_Depth.csv')
X2 = df[['Laser Power [W]', 'Scanning Speed [mm/s]', 'Layer Thickness [um]', 'Spot Size [um]', 'Porosity [%]']].values
y2 = df['Max Melt Pool Depth [um]'].values

# Create a Random Forest Regressor
Bagging_regressor = BaggingRegressor(base_estimator= DecisionTreeRegressor(),n_estimators=200, random_state=42)

Bagging_regressor.fit(X,y)

val_data = [[200,300,70,100,50]]
prediction_BR_depth = Bagging_regressor.predict(val_data)

# print message
print("depth BR: ")
print(prediction_BR_depth)
