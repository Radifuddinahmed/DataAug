import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import get_cmap
import numpy as np

var='depth'
df_ET = pd.read_csv(f'LPBF_ML_Model_Prediction_Width_ET.csv')
df_GP = pd.read_csv(f'LPBF_ML_Model_Prediction_Depth_GP.csv')

# Read data from CSV file
#df = pd.read_csv('width_new_model_comparison_results_20231214192952.csv')  # Replace with the actual path to your CSV file

# Extract model names and error values
actual_0 = df_ET['Actual'].tolist()
predicted_0 = df_ET['Predicted'].tolist()
serialNumber = df_ET['Number'].tolist()


# Extract model names and error values
actual_1 = df_GP['Actual'].tolist()
predicted_1 = df_GP['Predicted'].tolist()


cmap = get_cmap('Set3')

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 7))

# Plot Mean Absolute Error
axes[0].plot(serialNumber, actual_0, label='Actual')
axes[0].plot(serialNumber, predicted_0, label='Predicted')
# axes[0].set_title('Mean Absolute Error (MAE)')
axes[0].set_ylabel('Max Melt-Pool Width [um]')
axes[0].set_xlabel('Sample Number')


# Plot Root Mean Squared Error
axes[1].plot(serialNumber, actual_1, label='Actual')
axes[1].plot(serialNumber, predicted_1, label='Precicted')
# axes[1].set_title('Root Mean Squared Error (RMSE)')
axes[1].set_ylabel('Max Melt-Pool Depth [um]')
axes[1].set_xlabel('Sample Number')
#axes[1].set_ylabel('RMSE')




# # Set x-axis label at the bottom center
# fig.text(0.5, 0.05, 'Sample Number', ha='center', va='center')




# Add legend to the last subplot
axes[0].legend(loc='upper center', bbox_to_anchor=(0.1, 1.0), ncol=1)

# Add legend to the last subplot
axes[1].legend(loc='upper center', bbox_to_anchor=(0.1, 1.0), ncol=1)






# Save the plot as a PNG file
plt.savefig(f'LPBF_Plot_scatter_Fig8_Pred.png', bbox_inches='tight', dpi = 1200)

# # Show the plot
# plt.show()