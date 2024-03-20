import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import get_cmap
import numpy as np

var='depth'
df_nEstim = pd.read_csv(f'LPBF_HP_ExtraTrees_results_Width_nEstim_plot.csv')
df_maxDepth = pd.read_csv(f'LPBF_HP_ExtraTrees_results_Width_maxDepth_plot.csv')
df_minSamSplit = pd.read_csv(f'LPBF_HP_ExtraTrees_results_Width_minSamSplit_plot.csv')
df_minSamLeaf = pd.read_csv(f'LPBF_HP_ExtraTrees_results_Width_minSamLeaf_plot.csv')
# Read data from CSV file
#df = pd.read_csv('width_new_model_comparison_results_20231214192952.csv')  # Replace with the actual path to your CSV file

# Extract model names and error values
param = df_nEstim['n_estimators'].tolist()
mae_values_0 = df_nEstim['MAE'].tolist()
rmse_values_0 = df_nEstim['RMSE'].tolist()
rrse_values_0 = df_nEstim['RRSE'].tolist()
rae_values_0 = df_nEstim['RAE'].tolist()

# Extract model names and error values
param = df_maxDepth['max_depth'].tolist()
mae_values_1 = df_maxDepth['MAE'].tolist()
rmse_values_1 = df_maxDepth['RMSE'].tolist()
rrse_values_1 = df_maxDepth['RRSE'].tolist()
rae_values_1 = df_maxDepth['RAE'].tolist()

# Extract model names and error values
param_2 = df_minSamSplit['min_samples_split'].tolist()
mae_values_2 = df_minSamSplit['MAE'].tolist()
rmse_values_2 = df_minSamSplit['RMSE'].tolist()
rrse_values_2 = df_minSamSplit['RRSE'].tolist()
rae_values_2 = df_minSamSplit['RAE'].tolist()

# Extract model names and error values
param = df_minSamLeaf['min_samples_leaf'].tolist()
mae_values_3 = df_minSamLeaf['MAE'].tolist()
rmse_values_3 = df_minSamLeaf['RMSE'].tolist()
rrse_values_3 = df_minSamLeaf['RRSE'].tolist()
rae_values_3 = df_minSamLeaf['RAE'].tolist()

#correlation_values = df['R-squared'].tolist()


# Define colors for each model
#colors = plt.cm.viridis(np.linspace(0, 1, len(models)))

cmap = get_cmap('Set3')

fontSize = None

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8)) #(WIDHT, HEIGHT)

# Plot Mean Absolute Error
axes[0,0].plot(param, mae_values_0, label='n_estimators')
axes[0,0].plot(param, mae_values_1, label='max_depth')
axes[0,0].plot(param_2, mae_values_2, label='min_samples_split')
axes[0,0].plot(param, mae_values_3, label='min_samples_leaf')
axes[0,0].set_title('Mean Absolute Error (MAE)', fontsize= fontSize)
axes[0,0].set_ylabel('Error Values',  fontsize= fontSize)


# Plot Root Mean Squared Error
axes[0,1].plot(param, rmse_values_0, label='n_estimators')
axes[0,1].plot(param, rmse_values_1, label='max_depth')
axes[0,1].plot(param_2, rmse_values_2, label='min_samples_split')
axes[0,1].plot(param, rmse_values_3, label='min_samples_leaf')
axes[0,1].set_title('Root Mean Squared Error (RMSE)', fontsize= fontSize)
#axes[1].set_ylabel('RMSE')


# Plot Relative Root Squared Error
axes[1,0].plot(param, rrse_values_0, label='n_estimators')
axes[1,0].plot(param, rrse_values_1, label='max_depth')
axes[1,0].plot(param_2, rrse_values_2, label='min_samples_split')
axes[1,0].plot(param, rrse_values_3, label='min_samples_leaf')
axes[1,0].set_title('Relative Root Squared Error (RRSE)', fontsize= fontSize)
axes[1,0].set_ylabel('Error Values',  fontsize= fontSize)
#axes[2].set_ylabel('RRSE')


# Plot Relative Absolute Error
axes[1,1].plot(param, rae_values_0, label='n_estimators')
axes[1,1].plot(param, rae_values_1, label='max_depth')
axes[1,1].plot(param_2, rae_values_2, label='min_samples_split')
axes[1,1].plot(param, rae_values_3, label='min_samples_leaf')
axes[1,1].set_title('Relative Absolute Error (RAE)', fontsize= fontSize)
# axes[3].set_xlabel('Parameter Values', loc='center')

# Set x-axis label at the bottom center
fig.text(0.5, 0.07 , 'Parameter Values', ha='center', va='center', fontsize= fontSize)
# # Set y-axis label at the bottom center
# fig.text(0.5, 0.08 , 'Parameter Values', rotation=90, ha='center', va='center', fontsize= 12)
# # # Set y-axis label at the bottom center
# # plt.ylabel('Y-axis Label', rotation=90, ha='center', va='center')  # Rotate and position y label



# Add legend to the last subplot
axes[0,0].legend(loc='upper center', bbox_to_anchor=(1.0, 1.20), ncol=4, fontsize= fontSize )







# Save the plot as a PNG file
plt.savefig(f'LPBF_Plot_scatter_fig5.png', bbox_inches='tight', dpi = 600)

# # Show the plot
# plt.show()