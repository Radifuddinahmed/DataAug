import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor


from sklearn.ensemble import RandomForestRegressor # for building the model

df = pd.read_csv("filtered_dataset_no_outlier.csv")
X = df[['Tool Rotational Speed (RPM)', 'Translational Speed (mm/min)', 'Axial Force (KN)']].values
y = df['Ultimate Tensile Trength (MPa)'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestRegressor(n_estimators=20,  random_state=0)
rf_model.fit(X_train, y_train)
Y_pred =  rf_model.predict(X_test)
print(rf_model.score(X_test, y_test))
#print(Y_pred)

#k-fold Cross Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf_model, X, y, cv=10)
print(scores)
print(scores.mean())


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
mse = mean_squared_error(y_test, Y_pred)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

# Calculate the RAE
rae = np.mean(np.abs(y_test - Y_pred)) / np.mean(np.abs(y_test - np.mean(y_test)))
print("Relative Absolute Error (RAE):", rae)

# Calculate the RRSE
rrse = np.sqrt(np.sum((y_test - Y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))
print("Relative Root Squared Error (RRSE):", rrse)
