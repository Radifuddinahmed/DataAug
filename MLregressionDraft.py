##### Notebook properties for better display

# # Allow multiple outputs from single code chunk
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = 'all'

# Surpress Warnings
import warnings
warnings.filterwarnings("ignore")

##### Data Analysis
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format

import numpy as np

##### Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')

# Inline plotting
#%matplotlib inline

##### ML
## Cross validation
from sklearn.model_selection import train_test_split

## Linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

## Model evaluation tools
from yellowbrick.regressor import PredictionError, ResidualsPlot

# Load raw data
df = pd.read_csv("aksansh.csv")
X = df[['Tool Rotational Speed (RPM)', 'Translational Speed (mm/min)', 'Axial Force (KN)']].values
y = df['Ultimate Tensile Trength (MPa)'].values




from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.reshape(-1, 1))


# Create subplots for each feature
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))

# Plot feature 1
sns.scatterplot(ax=axes[0], x=X_scaled[:, 0], y=y_scaled[:, 0])
axes[0].set_xlabel('Scaled Feature 1')
axes[0].set_ylabel('Scaled Target')

# Plot feature 2
sns.scatterplot(ax=axes[1], x=X_scaled[:, 1], y=y_scaled[:, 0])
axes[1].set_xlabel('Scaled Feature 2')
axes[1].set_ylabel('Scaled Target')

# Plot feature 3
sns.scatterplot(ax=axes[2], x=X_scaled[:, 2], y=y_scaled[:, 0])
axes[2].set_xlabel('Scaled Feature 3')
axes[2].set_ylabel('Scaled Target')

plt.tight_layout()
plt.show()


num_additional_samples = 2

num_existing_samples = X_scaled.shape[0]  # Number of existing samples
num_total_samples = num_existing_samples + num_additional_samples  # Total number of samples after adding additional samples

# Randomly select indices with replacement from the existing dataset
random_indices = np.random.choice(num_existing_samples, size=num_additional_samples, replace=True)

# Generate additional samples using the randomly selected indices
additional_X = X_scaled[random_indices]
additional_y = y_scaled[random_indices]

# Concatenate the additional samples with the original dataset
new_X = np.concatenate((X_scaled, additional_X), axis=0)
new_y = np.concatenate((y_scaled, additional_y), axis=0)




# Create a DataFrame with the new dataset
data = {'feature1': new_X[:, 0],
        'feature2': new_X[:, 1],
        'feature3': new_X[:, 2],
        'target': new_y}

df_new = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df_new.to_csv('new_dataset.csv', index=False)
#
# #Gaussian Processes
#
# #Linear Regression - done
# #Multilayer Perception
# #Simple Linear Regression
# #SVM Regression
# #KNN
# #Random Forest
