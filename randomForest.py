# Importing the libraries
import numpy as np # for array operations
import pandas as pd # for working with DataFrames

import matplotlib.pyplot as plt # for data visualization


# scikit-learn modules
from sklearn.model_selection import train_test_split # for splitting the data
from sklearn.metrics import mean_squared_error # for calculating the cost function
from sklearn.ensemble import RandomForestRegressor # for building the model

df = pd.read_csv("FSW_Dataset_v4_test.csv")
X = df[['Tool Rotational Speed (RPM)', 'Translational Speed (mm/min)', 'Axial Force (KN)']].values
y = df['Ultimate Tensile Trength (MPa)'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestRegressor(n_estimators=11,  random_state=0)
rf_model.fit(X_train, y_train)
Y_pred =  rf_model.predict(X_test)

# Make data frame of above data
panda_data = pd.DataFrame(Y_pred)
print(panda_data)


# append data frame to CSV file
panda_data.to_csv('YPred.csv', mode='a', index=False, header=False)

# print message
print("Data appended successfully.")

#X_train.T, y_train.T

#calculate correlation coefficient
correlation_matrix = np.corrcoef(X_train.T, y_train.T)
correlation_coefficient = correlation_matrix[0, 1]  # Assuming X and y have shape (n_samples,)
print("Correlation coefficient:", correlation_coefficient)

#Calculate MAE
from sklearn.metrics import mean_absolute_error
# Calculate the Mean Absolute Error
mae = mean_absolute_error(y_test, Y_pred)
print("Mean Absolute Error:", mae)

#calculate RMSE
from sklearn.metrics import mean_squared_error
# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, Y_pred)
# Calculate the Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

# Calculate the RAE
rae = np.mean(np.abs(y_test - Y_pred)) / np.mean(np.abs(y_test - np.mean(y_test)))
print("Relative Absolute Error (RAE):", rae)

# Calculate the RRSE
rrse = np.sqrt(np.sum((y_test - Y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))
print("Relative Root Squared Error (RRSE):", rrse)


# c1 = [0,0,0]
# c2 = [0,0,0]
# c3 = [0,0,0]
# c4 = [0,0,0]
# opt_data = {'Tool Rotational Speed (RPM)': c1,
#         'Translational Speed (mm/min)': c2,
#         'Axial Force (KN)': c3,'Ultimate Tensile Trength (MPa)': c4 }
# opt = pd.DataFrame(opt_data)
#
# for rpm in np.arange(1,5,1):
#     for speed in np.arange(1, 5, 1):
#         for force in np.arange(1, 5, 1):
#             uts = (rf_model.predict([[rpm,speed,force]]))
#             print(rpm, speed, force, uts.item())
#             opt = pd.concat([opt, pd.DataFrame.from_records([{'Tool Rotational Speed (RPM)': rpm,
#                                                              'Translational Speed (mm/min)': speed,
#                                                               'Axial Force (KN)': force,
#                                                              'Ultimate Tensile Trength (MPa)': uts.item()}])])
#
#
# print(uts.item())
# opt.to_csv('rf_optim.csv', index=False)

