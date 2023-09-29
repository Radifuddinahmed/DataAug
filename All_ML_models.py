import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Assuming you have your data X and y
df = pd.read_csv("FSW_dataset_Original.csv")
X = df[['Tool Rotational Speed (RPM)', 'Translational Speed (mm/min)', 'Axial Force (KN)']].values
y = df['Ultimate Tensile Trength (MPa)'].values
k = 10

c1 = [0,0,0]
c2 = [0,0,0]
c3 = [0,0,0]
c4 = [0,0,0]
c5 = [0,0,0]
c6 = [0,0,0]
opt_data = {'Model': c1,
            'Correlation Coefficient': c2,
            'MAE': c3,
            'RMSE': c4,
            'RAE': c5,
            'RRSE':c6 }
opt = pd.DataFrame(opt_data)

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

opt = pd.concat([opt, pd.DataFrame.from_records([{'Model': "gaussian process",
                                                'Correlation Coefficient': correlation_coefficient,
                                                  'MAE': (mae_scores[-1]),
                                                  'RMSE': (rmse_scores[-1]),
                                                  'RAE': relative_absolute_error,
                                                  'RRSE':root_relative_squared_error }])])



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

opt = pd.concat([opt, pd.DataFrame.from_records([{'Model': "Linear Regression",
                                                'Correlation Coefficient': correlation_coefficient,
                                                  'MAE': (mae_scores[-1]),
                                                  'RMSE': (rmse_scores[-1]),
                                                  'RAE': relative_absolute_error,
                                                  'RRSE':root_relative_squared_error }])])


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

opt = pd.concat([opt, pd.DataFrame.from_records([{'Model': "Polynomial Regression",
                                                'Correlation Coefficient': correlation_coefficient,
                                                  'MAE': (mae_scores[-1]),
                                                  'RMSE': (rmse_scores[-1]),
                                                  'RAE': relative_absolute_error,
                                                  'RRSE':root_relative_squared_error }])])



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


opt = pd.concat([opt, pd.DataFrame.from_records([{'Model': "SVM Regression",
                                                'Correlation Coefficient': correlation_coefficient,
                                                  'MAE': (mae_scores[-1]),
                                                  'RMSE': (rmse_scores[-1]),
                                                  'RAE': relative_absolute_error,
                                                  'RRSE':root_relative_squared_error }])])




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

opt = pd.concat([opt, pd.DataFrame.from_records([{'Model': "KNN",
                                                'Correlation Coefficient': correlation_coefficient,
                                                  'MAE': (mae_scores[-1]),
                                                  'RMSE': (rmse_scores[-1]),
                                                  'RAE': relative_absolute_error,
                                                  'RRSE':root_relative_squared_error }])])




###### Random Forest Regressor #######
from sklearn.ensemble import RandomForestRegressor # for building the model
# Assuming you have your data X and y
# df = pd.read_csv("filtered_dataset_no_outlier_polynomial.csv")
# X = df[['Tool Rotational Speed (RPM)', 'Translational Speed (mm/min)', 'Axial Force (KN)']].values
# y = df['Ultimate Tensile Trength (MPa)'].values
k = 10
model_rf = RandomForestRegressor(n_estimators=20,  random_state=0)
print("Random Forest Regressor \n")

# Correlation Coefficient
predicted = cross_val_predict(model_rf, X, y, cv=k)
correlation_coefficient = -r2_score(y, predicted)
print("Correlation Coefficient (R-squared):", correlation_coefficient)

#Mean Absolute Error
mae_scores = -cross_val_score(model_rf, X, y, cv=k, scoring='neg_mean_absolute_error')
print("Mean Absolute Error (MAE):", np.mean(mae_scores))


# Root Mean Squared Error
mse_scores = -cross_val_score(model_rf, X, y, cv=k, scoring='neg_mean_squared_error')
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

print(type(mae_scores))
print("size:", mae_scores.size)
print("shape:", mae_scores.shape)
print("dimension:", mae_scores.ndim)
print("datatype:", mae_scores.dtype)

print(type(rmse_scores))






opt = pd.concat([opt, pd.DataFrame.from_records([{'Model': "Random Forest",
                                                'Correlation Coefficient': correlation_coefficient,
                                                  'MAE': (mae_scores[-1]),
                                                  'RMSE': (rmse_scores[-1]),
                                                  'RAE': relative_absolute_error,
                                                  'RRSE':root_relative_squared_error }])])


##### Multilayer perception #####

from sklearn.neural_network import MLPRegressor

model_nn = MLPRegressor(
    activation='relu',
    hidden_layer_sizes=(10, 100),
    alpha=0.001,
    random_state=20,
    early_stopping=False
)


print("Multi Layer Perception Regressor \n")

# Correlation Coefficient
predicted = cross_val_predict(model_nn, X, y, cv=k)
correlation_coefficient = -r2_score(y, predicted)
print("Correlation Coefficient (R-squared):", correlation_coefficient)

#Mean Absolute Error
mae_scores = -cross_val_score(model_nn, X, y, cv=k, scoring='neg_mean_absolute_error')
print("Mean Absolute Error (MAE):", np.mean(mae_scores))


# Root Mean Squared Error
mse_scores = -cross_val_score(model_nn, X, y, cv=k, scoring='neg_mean_squared_error')
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

opt = pd.concat([opt, pd.DataFrame.from_records([{'Model': "Multilayer Perception",
                                                'Correlation Coefficient': correlation_coefficient,
                                                  'MAE': (mae_scores),
                                                  'RMSE': (rmse_scores),
                                                  'RAE': relative_absolute_error,
                                                  'RRSE':root_relative_squared_error }])])



opt.to_csv('All_ML_models_Results.csv', index=False)