import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np

# Load the dataset
df = pd.read_csv('FSW_Dataset2.csv')


# Split the dataset into input features and target variable
X = df.drop('strength', axis = 1)
# X = df['rpm', 'speed', 'force']
y = df['strength']

# #X = df.drop('Ultimate Tensile Strength (MPa)', axis=1)
# #y = df["Ultimate Tensile Strength (MPa)"]

# Normalize the input features to have zero mean and unit variance
X = (X - X.mean()) / X.std()

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest regression model on the training set
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Define the parameter ranges for sensitivity analysis using Saltelli sampling
problem = {
    'num_vars': 3,
    'names': ['Tool Rotational Speed (RPM)', 'Translational Speed (mm/min)', 'Axial Force (KN)'],
    'bounds': [[X_train.min().min(), X_train.max().max()]] * 3
}
param_values = saltelli.sample(problem, 1000)


# Evaluate the model with the sampled parameter values
Y = model.predict(param_values)

# Compute the first-order and total sensitivity indices using Sobol analysis
Si = sobol.analyze(problem, Y)

# Print the sensitivity indices for each input feature
print("First-order indices:")
for i, name in enumerate(problem['names']):
    print(f"{name}: {Si['S1'][i]}")

print("Total indices:")
for i, name in enumerate(problem['names']):
    print(f"{name}: {Si['ST'][i]}")

    # Create a dictionary with the new dataset
    data = {'Tool Rotational Speed (RPM)': new_X[:, 0],
            'Translational Speed (mm/min)': new_X[:, 1],
            'Axial Force (KN)': new_X[:, 2],
            'Ultimate Tensile Trength (MPa)': new_y}

    # Ensure new_X is a 2-dimensional array
    if len(new_X.shape) == 1:
        new_X = new_X.reshape(-1, 1)

    # Create the DataFrame
    df_new = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df_new.to_csv('new_dataset.csv', index=False)