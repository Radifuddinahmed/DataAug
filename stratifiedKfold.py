import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression  # You can replace this with any regression model you prefer
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load your dataset into a pandas DataFrame (Replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv('FSW_Dataset_v4_test.csv')

# Split the data into features (X) and target variable (y)

X = data.drop('Ultimate Tensile Trength (MPa)', axis=1)  # Replace 'target_column' with the column name of the target variable
y = data['Ultimate Tensile Trength (MPa)']

# Create the regression model (You can use any other regression model instead of LinearRegression)
regression_model = LinearRegression()

# Set up the stratified k-fold cross-validation
num_folds = 5  # You can adjust the number of folds as per your requirement
stratified_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Initialize lists to store evaluation metrics for each fold
mae_scores = []
mse_scores = []

# Perform stratified k-fold cross-validation
for train_index, test_index in stratified_kfold.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the model on the training data
    regression_model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = regression_model.predict(X_test)

    # Evaluate the model using mean absolute error and mean squared error
    mae_scores.append(mean_absolute_error(y_test, y_pred))
    mse_scores.append(mean_squared_error(y_test, y_pred))

# Calculate the average metrics across all folds
average_mae = sum(mae_scores) / len(mae_scores)
average_mse = sum(mse_scores) / len(mse_scores)

print(f"Average MAE: {average_mae}")
print(f"Average MSE: {average_mse}")


#calculate correlation coefficient
correlation_matrix = np.corrcoef(X_train.T, y_train.T)
correlation_coefficient = correlation_matrix[0, 1]  # Assuming X and y have shape (n_samples,)
print("Correlation coefficient:", correlation_coefficient)

#Calculate MAE
from sklearn.metrics import mean_absolute_error
# Calculate the Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

#calculate RMSE
from sklearn.metrics import mean_squared_error
# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
# Calculate the Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

# Calculate the RAE
rae = np.mean(np.abs(y_test - y_pred)) / np.mean(np.abs(y_test - np.mean(y_test)))
print("Relative Absolute Error (RAE):", rae)

# Calculate the RRSE
rrse = np.sqrt(np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))
print("Relative Root Squared Error (RRSE):", rrse)
