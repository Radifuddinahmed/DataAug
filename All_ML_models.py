import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Assuming you have your data X and y
df = pd.read_csv("filtered_dataset_no_outlier_polynomial.csv")
X = df[['Tool Rotational Speed (RPM)', 'Translational Speed (mm/min)', 'Axial Force (KN)']].values
y = df['Ultimate Tensile Trength (MPa)'].values
k = 10

# Gaussian Process Regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

kernel = DotProduct() + WhiteKernel()
model = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X, y)
print("Gaussian Regression \n")

# Correlation Coefficient
predicted = cross_val_predict(model, X, y, cv=k)
correlation_coefficient = -r2_score(y, predicted)
print("Correlation Coefficient (R-squared):", correlation_coefficient)

#Mean Absolute Error
mae_scores = -cross_val_score(model, X, y, cv=k, scoring='neg_mean_absolute_error')
print("Mean Absolute Error (MAE):", np.mean(mae_scores))

# Root Mean Squared Error
mse_scores = -cross_val_score(model, X, y, cv=k, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(mse_scores)
print("Root Mean Squared Error (RMSE):", np.mean(rmse_scores))

# Relative Absolute Error
absolute_errors = np.abs(y - predicted)
# Calculate the mean absolute error
mean_absolute_error = np.mean(absolute_errors)
# Calculate the mean of the true values
mean_true_values = np.mean(y)
# Calculate the relative absolute error
relative_absolute_error = mean_absolute_error / mean_true_values
# Print the relative absolute error
print("Relative Absolute Error (RAE):", relative_absolute_error)

# Calculate the squared errors
squared_errors = np.square(y - predicted)
# Calculate the mean squared error
mean_squared_error = np.mean(squared_errors)
# Calculate the mean of the true values
mean_true_values = np.mean(y)
# Calculate the root relative squared error
root_relative_squared_error = np.sqrt(mean_squared_error) / mean_true_values
# Print the root relative squared error
print("Root Relative Squared Error (RRSE):", root_relative_squared_error)
print("\n")


# Linear Regression
model = LinearRegression()
print("linear Regression \n")

# Correlation Coefficient
predicted = cross_val_predict(model, X, y, cv=k)
correlation_coefficient = -r2_score(y, predicted)
print("Correlation Coefficient (R-squared):", correlation_coefficient)

#Mean Absolute Error
mae_scores = -cross_val_score(model, X, y, cv=k, scoring='neg_mean_absolute_error')
print("Mean Absolute Error (MAE):", np.mean(mae_scores))

# Root Mean Squared Error
mse_scores = -cross_val_score(model, X, y, cv=k, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(mse_scores)
print("Root Mean Squared Error (RMSE):", np.mean(rmse_scores))

# Relative Absolute Error
absolute_errors = np.abs(y - predicted)
# Calculate the mean absolute error
mean_absolute_error = np.mean(absolute_errors)
# Calculate the mean of the true values
mean_true_values = np.mean(y)
# Calculate the relative absolute error
relative_absolute_error = mean_absolute_error / mean_true_values
# Print the relative absolute error
print("Relative Absolute Error (RAE):", relative_absolute_error)

# Calculate the squared errors
squared_errors = np.square(y - predicted)
# Calculate the mean squared error
mean_squared_error = np.mean(squared_errors)
# Calculate the mean of the true values
mean_true_values = np.mean(y)
# Calculate the root relative squared error
root_relative_squared_error = np.sqrt(mean_squared_error) / mean_true_values
# Print the root relative squared error
print("Root Relative Squared Error (RRSE):", root_relative_squared_error)
print("\n")


#polynomial regression  - error
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(X.reshape(-1, 1))
from sklearn.linear_model import LinearRegression
poly_reg_model = LinearRegression()
# poly_reg_model.fit(poly_features, y)
# y_predicted = poly_reg_model.predict(poly_features)
# print(y_predicted)

# Correlation Coefficient
predicted = cross_val_predict(poly_reg_model, X, y, cv=k)
correlation_coefficient = -r2_score(y, predicted)
print("Correlation Coefficient (R-squared):", correlation_coefficient)

#Mean Absolute Error
mae_scores = -cross_val_score(model, X, y, cv=k, scoring='neg_mean_absolute_error')
print("Mean Absolute Error (MAE):", np.mean(mae_scores))

# Root Mean Squared Error
mse_scores = -cross_val_score(model, X, y, cv=k, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(mse_scores)
print("Root Mean Squared Error (RMSE):", np.mean(rmse_scores))

# Relative Absolute Error
absolute_errors = np.abs(y - predicted)
# Calculate the mean absolute error
mean_absolute_error = np.mean(absolute_errors)
# Calculate the mean of the true values
mean_true_values = np.mean(y)
# Calculate the relative absolute error
relative_absolute_error = mean_absolute_error / mean_true_values
# Print the relative absolute error
print("Relative Absolute Error (RAE):", relative_absolute_error)

# Calculate the squared errors
squared_errors = np.square(y - predicted)
# Calculate the mean squared error
mean_squared_error = np.mean(squared_errors)
# Calculate the mean of the true values
mean_true_values = np.mean(y)
# Calculate the root relative squared error
root_relative_squared_error = np.sqrt(mean_squared_error) / mean_true_values
# Print the root relative squared error
print("Root Relative Squared Error (RRSE):", root_relative_squared_error)
print("\n")

#SVM Regression
from sklearn.svm import SVR

model_SVR = SVR(kernel='linear', C=1)
print("Support Vector Machine \n")

# Correlation Coefficient
predicted = cross_val_predict(model_SVR, X, y, cv=k)
correlation_coefficient = -r2_score(y, predicted)
print("Correlation Coefficient (R-squared):", correlation_coefficient)

#Mean Absolute Error
mae_scores = -cross_val_score(model, X, y, cv=k, scoring='neg_mean_absolute_error')
print("Mean Absolute Error (MAE):", np.mean(mae_scores))


# Root Mean Squared Error
mse_scores = -cross_val_score(model, X, y, cv=k, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(mse_scores)
print("Root Mean Squared Error (RMSE):", np.mean(rmse_scores))

# Relative Absolute Error
absolute_errors = np.abs(y - predicted)
# Calculate the mean absolute error
mean_absolute_error = np.mean(absolute_errors)
# Calculate the mean of the true values
mean_true_values = np.mean(y)
# Calculate the relative absolute error
relative_absolute_error = mean_absolute_error / mean_true_values
# Print the relative absolute error
print("Relative Absolute Error (RAE):", relative_absolute_error)

# Calculate the squared errors
squared_errors = np.square(y - predicted)
# Calculate the mean squared error
mean_squared_error = np.mean(squared_errors)
# Calculate the mean of the true values
mean_true_values = np.mean(y)
# Calculate the root relative squared error
root_relative_squared_error = np.sqrt(mean_squared_error) / mean_true_values
# Print the root relative squared error
print("Root Relative Squared Error (RRSE):", root_relative_squared_error)
print("\n")


#KNN
from sklearn.neighbors import KNeighborsRegressor
model_KNN = KNeighborsRegressor(n_neighbors=2)
print("KNN Regressor \n")

# Correlation Coefficient
predicted = cross_val_predict(model_KNN, X, y, cv=k)
correlation_coefficient = -r2_score(y, predicted)
print("Correlation Coefficient (R-squared):", correlation_coefficient)

#Mean Absolute Error
mae_scores = -cross_val_score(model, X, y, cv=k, scoring='neg_mean_absolute_error')
print("Mean Absolute Error (MAE):", np.mean(mae_scores))


# Root Mean Squared Error
mse_scores = -cross_val_score(model, X, y, cv=k, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(mse_scores)
print("Root Mean Squared Error (RMSE):", np.mean(rmse_scores))

# Relative Absolute Error
absolute_errors = np.abs(y - predicted)
# Calculate the mean absolute error
mean_absolute_error = np.mean(absolute_errors)
# Calculate the mean of the true values
mean_true_values = np.mean(y)
# Calculate the relative absolute error
relative_absolute_error = mean_absolute_error / mean_true_values
# Print the relative absolute error
print("Relative Absolute Error (RAE):", relative_absolute_error)

# Calculate the squared errors
squared_errors = np.square(y - predicted)
# Calculate the mean squared error
mean_squared_error = np.mean(squared_errors)
# Calculate the mean of the true values
mean_true_values = np.mean(y)
# Calculate the root relative squared error
root_relative_squared_error = np.sqrt(mean_squared_error) / mean_true_values
# Print the root relative squared error
print("Root Relative Squared Error (RRSE):", root_relative_squared_error)
print("\n")


# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor # for building the model
# Assuming you have your data X and y
df = pd.read_csv("filtered_dataset_no_outlier_polynomial.csv")
X = df[['Tool Rotational Speed (RPM)', 'Translational Speed (mm/min)', 'Axial Force (KN)']].values
y = df['Ultimate Tensile Trength (MPa)'].values
k = 10
model_rf = RandomForestRegressor(n_estimators=20,  random_state=0)
print("Random Forest Regressor \n")

# Correlation Coefficient
predicted = cross_val_predict(model_rf, X, y, cv=k)
correlation_coefficient = -r2_score(y, predicted)
print("Correlation Coefficient (R-squared):", correlation_coefficient)

#Mean Absolute Error
mae_scores = -cross_val_score(model, X, y, cv=k, scoring='neg_mean_absolute_error')
print("Mean Absolute Error (MAE):", np.mean(mae_scores))


# Root Mean Squared Error
mse_scores = -cross_val_score(model, X, y, cv=k, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(mse_scores)
print("Root Mean Squared Error (RMSE):", np.mean(rmse_scores))

# Relative Absolute Error
absolute_errors = np.abs(y - predicted)
# Calculate the mean absolute error
mean_absolute_error = np.mean(absolute_errors)
# Calculate the mean of the true values
mean_true_values = np.mean(y)
# Calculate the relative absolute error
relative_absolute_error = mean_absolute_error / mean_true_values
# Print the relative absolute error
print("Relative Absolute Error (RAE):", relative_absolute_error)

# Calculate the squared errors
squared_errors = np.square(y - predicted)
# Calculate the mean squared error
mean_squared_error = np.mean(squared_errors)
# Calculate the mean of the true values
mean_true_values = np.mean(y)
# Calculate the root relative squared error
root_relative_squared_error = np.sqrt(mean_squared_error) / mean_true_values
# Print the root relative squared error
print("Root Relative Squared Error (RRSE):", root_relative_squared_error)
print("\n")
