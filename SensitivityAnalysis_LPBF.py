import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np

# Load the dataset as an example
df = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\LPBF_Dataset_Combined_v2_python_Depth.csv')
X = df[['Laser Power [W]', 'Scanning Speed [mm/s]', 'Layer Thickness [um]', 'Spot Size [um]', 'Porosity [%]']].values
y = df['Max Melt Pool Depth [um]'].values
# # Normalize the input features to have zero mean and unit variance
# X = (X - X.mean()) / X.std() # this normalization does not have 0 to 1 as range

# # Perform Min-Max scaling to the range [0, 1]
# min_val = 0
# max_val = 1
#
# X = (X - np.min(X)) / (np.max(X) - np.min(X)) * (max_val - min_val) + min_val
#
# print(X)
#
# # Create the DataFrame
# df_new = pd.DataFrame(X)
#
# # Save the DataFrame to a CSV file
# df_new.to_csv('SensitivityAnalysis_LPBF_1.csv', index=False)

# Train a random forest regression model on the training set
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Define the parameter ranges for sensitivity analysis using Saltelli sampling
problem = {
    'num_vars': 5,
    'names': ['Power', 'Scanning Speed', 'Layer Thickness', 'Spot Size', 'Porosity'],
    'bounds': [[0, 1]] * 5 #here if -1 and 1 is taken as range then all parameters have values, if 0 and 1 is taken then values come out as zero
}
param_values = saltelli.sample(problem, 1024)

# Evaluate the model with the sampled parameter values
Y = model.predict(param_values)

# Compute the first-order and total sensitivity indices using Sobol analysis
Si = sobol.analyze(problem, Y)

# Print the sensitivity indices for each input feature
print("First-order indices:")
for i, name in enumerate(problem['names']):
    print(f"{name}: {Si['S1'][i]} ")

print("\nTotal indices:")
for i, name in enumerate(problem['names']):
    print(f"{name}: {Si['ST'][i]}")


