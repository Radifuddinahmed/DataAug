import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import get_cmap
import numpy as np

var='depth'
df = pd.read_csv(f'LPBF_ML_Model_Evaluation_HpTuning_Width.csv')


# Extract model names and error values
models = df['Model'].tolist()
mae_values = df['MAE'].tolist()
rmse_values = df['RMSE'].tolist()
rrse_values = df['RRSE'].tolist()
rae_values = df['RAE'].tolist()
correlation_values = df['Correlation Coefficient'].tolist()

# Define colors for each model
cmap = get_cmap('Set3')
colors = cmap(np.linspace(0, 1, len(models)))

# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 4))


# Plot Mean Absolute Error
for i, color in zip(range(len(models)), colors):
    axes[0].bar(i, mae_values[i], color=color, label=f'{models[i]}')
axes[0].set_title('Mean Absolute Error (MAE)')
axes[0].set_ylabel('Error Values')
#axes[0].set_ylabel('MAE')


# Plot Root Mean Squared Error
for i, color in zip(range(len(models)), colors):
    axes[1].bar(i, rmse_values[i], color=color)
axes[1].set_title('Root Mean Squared Error (RMSE)')
#axes[1].set_ylabel('RMSE')


# Plot Relative Root Squared Error
for i, color in zip(range(len(models)), colors):
    axes[2].bar(i, rrse_values[i], color=color)
axes[2].set_title('Relative Root Squared Error (RRSE)')
#axes[2].set_ylabel('RRSE')


# Plot Relative Absolute Error
for i, color in zip(range(len(models)), colors):
    axes[3].bar(i, rae_values[i], color=color)
axes[3].set_title('Relative Absolute Error (RAE)')
#axes[3].set_ylabel('RAE')

# Plot Correlation Coefficient
for i, color in zip(range(len(models)), colors):
    axes[4].bar(i, correlation_values[i], color=color, label=f'{models[i]}')

axes[4].set_title('Correlation Coefficient')
#axes[4].set_ylabel('Correlation')



# Add legend to the last subplot
axes[0].legend(loc='upper center', bbox_to_anchor=(2.8, 1.30), ncol=8)
fig.text(0.5, 0.05, 'Machine Learning Model', ha='center', va='center')
# plt.legend(loc='upper right')
# plt.legend(title='Legend Title', fontsize='medium', facecolor='lightgray')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


# Remove x-axis ticks and labels for all subplots
for ax in axes:
    ax.set_xticks([])
    ax.set_xticklabels([])

# Adjust layout for better spacing
#plt.tight_layout()



# Save the plot as a PNG file
plt.savefig(f'LPBF_Plot_barChart_Fig8.png', bbox_inches='tight')

# Show the plot
plt.show()

