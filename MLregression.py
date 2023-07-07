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
df = pd.read_csv("FSW_Dataset_v4_test.csv")
X = df[['Tool Rotational Speed (RPM)', 'Translational Speed (mm/min)', 'Axial Force (KN)']].values
y = df['Ultimate Tensile Trength (MPa)'].values


fsw_corr_df = (
    df
    .corr()
    .round(2)
)

sns.heatmap(fsw_corr_df, cmap='vlag_r', vmin=-1, vmax=1)
#plt.show()

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)
print(X_train.shape)
print(y_train.shape)

# Initialize a linear regression model object
model = LinearRegression();

# Train the model with the training data and labels
model.fit(X_train, y_train);

print(model.intercept_)
print(model.coef_)



y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)



# View the results
df_new=pd.DataFrame(
    {
        'quality_score': y_test,
        'predicted_quality_score': np.round(y_test_pred, 2)
    }
)



# Save the DataFrame to a CSV file
df_new.to_csv('new_dataset.csv', index=False)

#calculate correlation coefficient
correlation_matrix = np.corrcoef(X_train.T, y_train.T)
correlation_coefficient = correlation_matrix[0, 1]  # Assuming X and y have shape (n_samples,)
print("Correlation coefficient:", correlation_coefficient)

#Calculate MAE
from sklearn.metrics import mean_absolute_error
# Calculate the Mean Absolute Error
mae = mean_absolute_error(y_test, y_test_pred)
print("Mean Absolute Error:", mae)

#calculate RMSE
from sklearn.metrics import mean_squared_error
# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_test_pred)
# Calculate the Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

# Calculate the RAE
rae = np.mean(np.abs(y_test - y_test_pred)) / np.mean(np.abs(y_test - np.mean(y_test)))
print("Relative Absolute Error (RAE):", rae)

# Calculate the RRSE
rrse = np.sqrt(np.sum((y_test - y_test_pred)**2) / np.sum((y_test - np.mean(y_test))**2))
print("Relative Root Squared Error (RRSE):", rrse)





#model Evaluation
fsw_pred = PredictionError(model, is_fitted=True, bestfit=False, alpha=0.4)

fsw_pred.score(X_test, y_test)

fsw_pred.show();

print(model.predict([[100,50,15]]))



c1 = [0,0,0]
c2 = [0,0,0]
c3 = [0,0,0]
c4 = [0,0,0]
opt_data = {'Tool Rotational Speed (RPM)': c1,
        'Translational Speed (mm/min)': c2,
        'Axial Force (KN)': c3,'Ultimate Tensile Trength (MPa)': c4 }
opt = pd.DataFrame(opt_data)

for rpm in np.arange(700,2023,1):
    for speed in np.arange(1.6, 157.5, 1.1):
        for force in np.arange(1.8, 10.4, 0.1):
            uts = model.predict([[rpm,speed,force]])
            print(rpm, speed, force, uts.item())
            opt = pd.concat([opt, pd.DataFrame.from_records([{'Tool Rotational Speed (RPM)': rpm,
                                                             'Translational Speed (mm/min)': speed,
                                                              'Axial Force (KN)': force,
                                                             'Ultimate Tensile Trength (MPa)': uts.item()}])])


opt.to_csv('rf_optimization.csv',encoding='utf-8', index=False,float_format='%.2f')

