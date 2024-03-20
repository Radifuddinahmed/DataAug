import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import get_cmap
import numpy as np

var='depth'
df = pd.read_csv(f'LPBF_Plot_barChart_fig9_Width_ET.csv')
# Read data from CSV file
#df = pd.read_csv('width_new_model_comparison_results_20231214192952.csv')  # Replace with the actual path to your CSV file

# Extract model names and error values
models = df['Method'].tolist()
mae_values = df['MAE'].tolist()
rmse_values = df['RMSE'].tolist()
rrse_values = df['RRSE'].tolist()
rae_values = df['RAE'].tolist()
correlation_values = df['r2'].tolist()

#correlation_values = df['R-squared'].tolist()


# Define colors for each model
cmap = get_cmap('Set3')
colors = cmap(np.linspace(0, 1, len(models)))

# Create subplots
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(9, 10))

fontSize = None


# Plot Mean Absolute Error
for i, color in zip(range(len(models)), colors):
    axes[0,0].bar(i, mae_values[i], color=color, label=f'{models[i]}')
axes[0, 0].tick_params(axis='x', which='both', bottom=False, top=False)  # Disable x-axis ticks and labels
axes[0, 0].set_xticklabels([])  # Remove x-axis labels
axes[0,0].set_title('Mean Absolute Error (MAE)', x = 1.1, fontsize= fontSize)


#axes[0].set_ylabel('MAE')
# Add legend to the last subplot
axes[0,0].legend(loc='upper center', bbox_to_anchor=(2.8, 1.20), ncol=4)


# Plot Root Mean Squared Error
for i, color in zip(range(len(models)), colors):
    axes[1,0].bar(i, rmse_values[i], color=color)
axes[1, 0].tick_params(axis='x', which='both', bottom=False, top=False)  # Disable x-axis ticks and labels
axes[1, 0].set_xticklabels([])  # Remove x-axis labels
axes[1,0].set_title('Root Mean Squared Error (RMSE)', x = 1.1, fontsize= fontSize)
#axes[1].set_ylabel('RMSE')


# Plot Relative Root Squared Error
for i, color in zip(range(len(models)), colors):
    axes[2,0].bar(i, rrse_values[i], color=color)
axes[2,0].tick_params(axis='x', which='both', bottom=False, top=False)  # Disable x-axis ticks and labels
axes[2,0].set_xticklabels([])  # Remove x-axis labels
axes[2,0].set_title('Relative Root Squared Error (RRSE)', x = 1.1, fontsize= fontSize)
# axes[2,0].set_ylabel('Performance Metric', fontsize= fontSize)

#axes[2].set_ylabel('RRSE')


# Plot Relative Absolute Error
for i, color in zip(range(len(models)), colors):
    axes[3,0].bar(i, rae_values[i], color=color)
axes[3,0].tick_params(axis='x', which='both', bottom=False, top=False)  # Disable x-axis ticks and labels
axes[3,0].set_xticklabels([])  # Remove x-axis labels
axes[3,0].set_title('Relative Absolute Error (RAE)', x = 1.1, fontsize= fontSize)
#axes[3].set_ylabel('RAE')

# Plot Correlation Coefficient
for i, color in zip(range(len(models)), colors):
    axes[4,0].bar(i, correlation_values[i], color=color, label=f'{models[i]}')
axes[4,0].tick_params(axis='x', which='both', bottom=False, top=False)  # Disable x-axis ticks and labels
axes[4,0].set_xticklabels([])  # Remove x-axis labels
axes[4,0].set_title('Correlation Coefficient', x = 1.1, fontsize= fontSize)
#axes[4].set_ylabel('Correlation')




# Add legend to the last subplot
axes[0,0].legend(loc='upper center', bbox_to_anchor=(1.1, 1.55), ncol=3, fontsize= fontSize)
# fig.text(0.5, 0.09, 'Machine Learning Model', ha='center', va='center', fontsize= fontSize)










df = pd.read_csv(f'LPBF_Plot_barChart_fig9_Depth_GP.csv')
# Read data from CSV file
#df = pd.read_csv('width_new_model_comparison_results_20231214192952.csv')  # Replace with the actual path to your CSV file

# Extract model names and error values
models = df['Method'].tolist()
mae_values = df['MAE'].tolist()
rmse_values = df['RMSE'].tolist()
rrse_values = df['RRSE'].tolist()
rae_values = df['RAE'].tolist()
correlation_values = df['r2'].tolist()


# Plot Mean Absolute Error
for i, color in zip(range(len(models)), colors):
    axes[0,1].bar(i, mae_values[i], color=color, label=f'{models[i]}')
axes[0, 1].tick_params(axis='x', which='both', bottom=False, top=False)  # Disable x-axis ticks and labels
axes[0, 1].set_xticklabels([])  # Remove x-axis labels
# axes[0,1].set_title('Mean Absolute Error (MAE)', fontsize= fontSize)

#axes[0].set_ylabel('MAE')


# Plot Root Mean Squared Error
for i, color in zip(range(len(models)), colors):
    axes[1,1].bar(i, rmse_values[i], color=color)
axes[1, 1].tick_params(axis='x', which='both', bottom=False, top=False)  # Disable x-axis ticks and labels
axes[1, 1].set_xticklabels([])  # Remove x-axis labels
# axes[1,1].set_title('Root Mean Squared Error (RMSE)', fontsize= fontSize)
#axes[1].set_ylabel('RMSE')


# Plot Relative Root Squared Error
for i, color in zip(range(len(models)), colors):
    axes[2,1].bar(i, rrse_values[i], color=color)
axes[2,1].tick_params(axis='x', which='both', bottom=False, top=False)  # Disable x-axis ticks and labels
axes[2,1].set_xticklabels([])  # Remove x-axis labels
# axes[2,1].set_title('Relative Root Squared Error (RRSE)', fontsize= fontSize)
# axes[2,1].set_ylabel('Performance Metric', fontsize= fontSize)
#axes[2].set_ylabel('RRSE')


# Plot Relative Absolute Error
for i, color in zip(range(len(models)), colors):
    axes[3,1].bar(i, rae_values[i], color=color)
axes[3,1].tick_params(axis='x', which='both', bottom=False, top=False)  # Disable x-axis ticks and labels
axes[3,1].set_xticklabels([])  # Remove x-axis labels
# axes[3,1].set_title('Relative Absolute Error (RAE)', fontsize= fontSize)
#axes[3].set_ylabel('RAE')

# Plot Correlation Coefficient
for i, color in zip(range(len(models)), colors):
    axes[4,1].bar(i, correlation_values[i], color=color, label=f'{models[i]}')
axes[4,1].tick_params(axis='x', which='both', bottom=False, top=False)  # Disable x-axis ticks and labels
axes[4,1].set_xticklabels([])  # Remove x-axis labels
# axes[4,1].set_title('Correlation Coefficient', fontsize= fontSize)
axes[4,1].set_ylim([-0.1, 0.95])
#axes[4].set_ylabel('Correlation')






# Save the plot as a PNG file
plt.savefig(f'LPBF_Plot_barChart_Fig9.png', bbox_inches='tight', dpi = 600)



















# import matplotlib.pyplot as plt
# import pandas as pd
# from matplotlib.cm import get_cmap
# import numpy as np
#
# var='depth'
# df = pd.read_csv(f'LPBF_Plot_barChart_fig9_Width_ET.csv')
# # Read data from CSV file
# #df = pd.read_csv('width_new_model_comparison_results_20231214192952.csv')  # Replace with the actual path to your CSV file
#
# # Extract model names and error values
# models = df['Method'].tolist()
# mae_values = df['MAE'].tolist()
# rmse_values = df['RMSE'].tolist()
# rrse_values = df['RRSE'].tolist()
# rae_values = df['RAE'].tolist()
# correlation_values = df['r2'].tolist()
#
# #correlation_values = df['R-squared'].tolist()
#
#
# # Define colors for each model
# cmap = get_cmap('Set3')
# colors = cmap(np.linspace(0, 1, len(models)))
#
# # Create subplots
# fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))
#
# # Plot Mean Absolute Error
# for i, color in zip(range(len(models)), colors):
#     axes[0].bar(i, mae_values[i], color=color, label=f'{models[i]}')
# axes[0].set_title('Mean Absolute Error (MAE)')
# axes[0].set_ylabel('Performance Metric')
# #axes[0].set_ylabel('MAE')
# # Add legend to the last subplot
# axes[0].legend(loc='upper center', bbox_to_anchor=(2.8, 1.20), ncol=4)
#
#
# # Plot Root Mean Squared Error
# for i, color in zip(range(len(models)), colors):
#     axes[1].bar(i, rmse_values[i], color=color)
# axes[1].set_title('Root Mean Squared Error (RMSE)')
# #axes[1].set_ylabel('RMSE')
#
#
# # Plot Relative Root Squared Error
# for i, color in zip(range(len(models)), colors):
#     axes[2].bar(i, rrse_values[i], color=color)
# axes[2].set_title('Relative Root Squared Error (RRSE)')
# #axes[2].set_ylabel('RRSE')
#
#
# # Plot Relative Absolute Error
# for i, color in zip(range(len(models)), colors):
#     axes[3].bar(i, rae_values[i], color=color)
# axes[3].set_title('Relative Absolute Error (RAE)')
# #axes[3].set_ylabel('RAE')
#
#
# # Plot Correlation Coefficient
# for i, color in zip(range(len(models)), colors):
#     axes[4].bar(i, correlation_values[i], color=color, label=f'{models[i]}')
# axes[4].set_title('Correlation Coefficient')
#
#
# fig.text(0.5, 0.05, 'ML Model', ha='center', va='center')
#
#
#
# # Remove x-axis ticks and labels for all subplots
# for ax in axes:
#     ax.set_xticks([])
#     ax.set_xticklabels([])
#
# # Adjust layout for better spacing
# #plt.tight_layout()
#
#
#
# # Save the plot as a PNG file
# plt.savefig(f'LPBF_Plot_barChart_Fig9_Width_ET.png', bbox_inches='tight')
#
# # Show the plot
# # plt.show()
#
#
# df = pd.read_csv(f'LPBF_Plot_barChart_fig9_Depth_GP.csv')
# # Read data from CSV file
# #df = pd.read_csv('width_new_model_comparison_results_20231214192952.csv')  # Replace with the actual path to your CSV file
#
# # Extract model names and error values
# models = df['Method'].tolist()
# mae_values = df['MAE'].tolist()
# rmse_values = df['RMSE'].tolist()
# rrse_values = df['RRSE'].tolist()
# rae_values = df['RAE'].tolist()
# correlation_values = df['r2'].tolist()
#
# #correlation_values = df['R-squared'].tolist()
#
#
# # Define colors for each model
# cmap = get_cmap('Set3')
# colors = cmap(np.linspace(0, 1, len(models)))
#
# # Create subplots
# fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))
#
# # Plot Mean Absolute Error
# for i, color in zip(range(len(models)), colors):
#     axes[0].bar(i, mae_values[i], color=color, label=f'{models[i]}')
# axes[0].set_title('Mean Absolute Error (MAE)')
# axes[0].set_ylabel('Performance Metric')
# #axes[0].set_ylabel('MAE')
# # Add legend to the last subplot
# axes[0].legend(loc='upper center', bbox_to_anchor=(2.8, 1.20), ncol=4)
#
#
# # Plot Root Mean Squared Error
# for i, color in zip(range(len(models)), colors):
#     axes[1].bar(i, rmse_values[i], color=color)
# axes[1].set_title('Root Mean Squared Error (RMSE)')
# #axes[1].set_ylabel('RMSE')
#
#
# # Plot Relative Root Squared Error
# for i, color in zip(range(len(models)), colors):
#     axes[2].bar(i, rrse_values[i], color=color)
# axes[2].set_title('Relative Root Squared Error (RRSE)')
# #axes[2].set_ylabel('RRSE')
#
#
# # Plot Relative Absolute Error
# for i, color in zip(range(len(models)), colors):
#     axes[3].bar(i, rae_values[i], color=color)
# axes[3].set_title('Relative Absolute Error (RAE)')
# #axes[3].set_ylabel('RAE')
#
# # Plot Correlation Coefficient
# for i, color in zip(range(len(models)), colors):
#     axes[4].bar(i, correlation_values[i], color=color, label=f'{models[i]}')
# axes[4].set_title('Correlation Coefficient')
# axes[4].set_ylim([-0.1, 0.95])
#
# fig.text(0.5, 0.05, 'ML Model', ha='center', va='center')
#
#
#
# # Remove x-axis ticks and labels for all subplots
# for ax in axes:
#     ax.set_xticks([])
#     ax.set_xticklabels([])
#
# # Adjust layout for better spacing
# #plt.tight_layout()
#
#
#
# # Save the plot as a PNG file
# plt.savefig(f'LPBF_Plot_barChart_Fig9_Depth_GP.png', bbox_inches='tight')