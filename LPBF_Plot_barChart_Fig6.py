import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import get_cmap
import numpy as np

var='depth'
df = pd.read_csv(f'LPBF_HP_GaussianProcess_results_depth_Plot.csv')


# Extract model names and error values
models = df['Kernel'].tolist()
mae_values = df['MAE'].tolist()
rmse_values = df['RMSE'].tolist()
rrse_values = df['RRSE'].tolist()
rae_values = df['RAE'].tolist()

# Define colors for each model
cmap = get_cmap('Set3')
colors = cmap(np.linspace(0, 1, len(models)))

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,9))
# Font size
fontSize = None

# Plot Mean Absolute Error
for i, color in zip(range(len(models)), colors):
    axes[0,0].bar(i, mae_values[i], color=color, label=f'{models[i]}')

axes[0,0].set_title('Mean Absolute Error (MAE)', fontsize= fontSize)
# axes[0,0].set_ylabel('Error Values', fontsize= fontSize)
axes[0, 0].set_xticklabels([])  # Remove x-axis labels
axes[0, 0].set_xticks([])

#axes[0].set_ylabel('MAE')


# Plot Root Mean Squared Error
for i, color in zip(range(len(models)), colors):
    axes[0,1].bar(i, rmse_values[i], color=color)
axes[0,1].set_title('Root Mean Squared Error (RMSE)', fontsize= fontSize)
axes[0, 1].set_xticklabels([])  # Remove x-axis labels
axes[0, 1].set_xticks([])
#axes[1].set_ylabel('RMSE')


# Plot Relative Root Squared Error
for i, color in zip(range(len(models)), colors):
    axes[1,0].bar(i, rrse_values[i], color=color)
axes[1,0].set_title('Relative Root Squared Error (RRSE)', fontsize= fontSize)
# axes[1,0].set_ylabel('Error Values', fontsize= fontSize)
axes[1, 0].set_xticklabels([])  # Remove x-axis labels
axes[1, 0].set_xticks([])
#axes[2].set_ylabel('RRSE')


# Plot Relative Absolute Error
for i, color in zip(range(len(models)), colors):
    axes[1,1].bar(i, rae_values[i], color=color)
axes[1,1].set_title('Relative Absolute Error (RAE)', fontsize= fontSize)
axes[1, 1].set_xticklabels([])  # Remove x-axis labels
axes[1, 1].set_xticks([])

#axes[3].set_ylabel('RAE')



# Add legend to the last subplot
axes[0,0].legend(loc='upper center', bbox_to_anchor=(1.10, 1.4), ncol=2)
# fig.text(0.5, 0.08 , 'Kernel', ha='center', va='center', fontsize= fontSize)
# plt.legend(loc='upper right')
# plt.legend(title='Legend Title', fontsize='medium', facecolor='lightgray')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


# # Remove x-axis ticks and labels for all subplots
# for ax in axes:
#     ax.set_xticks([])
#     ax.set_xticklabels([])
#
# # Adjust layout for better spacing
# #plt.tight_layout()



# Save the plot as a PNG file
plt.savefig(f'LPBF_Plot_barChart_Fig6.png', bbox_inches='tight', dpi = 600)

# # Show the plot
# plt.show()
