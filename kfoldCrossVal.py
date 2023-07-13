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


from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

# Assuming you have your data X and y
df = pd.read_csv("filtered_dataset_no_outlier.csv")
X = df[['Tool Rotational Speed (RPM)', 'Translational Speed (mm/min)', 'Axial Force (KN)']].values
y = df['Ultimate Tensile Trength (MPa)'].values
k = 10
# Create your regression model
model = LinearRegression()

# Perform k-fold cross-validation and calculate evaluation metrics
mse_scores = -cross_val_score(model, X, y, cv=k, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(mse_scores)
mae_scores = -cross_val_score(model, X, y, cv=k, scoring='neg_mean_absolute_error')


# Print the evaluation metrics

print("Root Mean Squared Error (RMSE):", np.mean(rmse_scores))
print("Mean Absolute Error (MAE):", np.mean(mae_scores))


# Perform k-fold cross-validation and calculate MAE
mae_scores = -cross_val_score(model, X, y, cv=k, scoring='neg_mean_absolute_error')

# Print the mean absolute error
print("Mean Absolute Error (MAE):", np.mean(mae_scores))

from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict
# Train your model and generate predictions using k-fold cross-validation
predicted = cross_val_predict(model, X, y, cv=k)
# Calculate the correlation coefficient (R-squared)
correlation_coefficient = -r2_score(y, predicted)
# Print the correlation coefficient
print("Correlation Coefficient (R-squared):", correlation_coefficient)



# Calculate the absolute errors
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