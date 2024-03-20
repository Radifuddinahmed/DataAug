import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.cm import get_cmap

var='Depth'
df = pd.read_csv(f'LPBF_ML_Model_Evaluation_BaseModel_{var}.csv')
df2 = pd.read_csv(f'Poly6{var}_base.csv') # different result - poly 6 with base models

#HP
df3 = pd.read_csv(f'LPBF_ML_Model_Evaluation_HpTuning_{var}.csv')
df4 = pd.read_csv(f'LPBF_ML_Model_Evaluation_Poly6_{var}.csv')

# Read data from CSV file
# df = pd.read_csv('width_new_model_comparison_results_20231214192952.csv')  # Replace with the actual path to your CSV file

# Extract model names and error values
models = df['Model'].tolist()

# BASE
mae_values = df['MAE'].tolist()
rmse_values = df['RMSE'].tolist()
rrse_values = df['RRSE'].tolist()
rae_values = df['RAE'].tolist()
correlation_values = df['Correlation Coefficient'].tolist()
# correlation_values = df['R-squared'].tolist()

# Poly Base
mae_values2 = df2['MAE'].tolist()
rmse_values2 = df2['RMSE'].tolist()
rrse_values2 = df2['RRSE'].tolist()
rae_values2 = df2['RAE'].tolist()
# correlation_values2 = df2['Correlation Coefficient'].tolist()
correlation_values2 = df2['R-squared'].tolist()

# HP BASE
mae_values3 = df3['MAE'].tolist()
rmse_values3 = df3['RMSE'].tolist()
rrse_values3 = df3['RRSE'].tolist()
rae_values3 = df3['RAE'].tolist()
correlation_values3 = df3['Correlation Coefficient'].tolist()

# HP POLY

mae_values4 = df4['MAE'].tolist()
rmse_values4 = df4['RMSE'].tolist()
rrse_values4 = df4['RRSE'].tolist()
rae_values4 = df4['RAE'].tolist()
correlation_values4 = df4['Correlation Coefficient'].tolist()

models_n = ['GP', 'LR', 'PR', 'SVR', 'KNN', 'MLP', 'RF', 'GB', 'AB', 'B', 'ET']

bar_width = 0.2  # Width of each bar
bar_space = 0.02

x = np.arange(len(models))  # X positions for the categories

# Define colors for each model
cmap = get_cmap('Set3')
colors = cmap(np.linspace(0, 1, len(models)))

colorA = [0.55294118, 0.82745098, 0.78039216, 1.        ]
colorB=[1.,         1.  ,       0.70196078 ,1.        ]
colorC=[0.98431373 ,0.50196078 ,0.44705882, 1.        ]
colorD=[0.99215686 ,0.70588235 ,0.38431373, 1.        ]
colorE=[0.70196078 ,0.87058824, 0.41176471, 1.        ]
colorF=[0.85098039, 0.85098039 ,0.85098039 ,1.        ]
colorG=[0.8  ,      0.92156863, 0.77254902, 1.        ]
colorH= [1.    ,     0.92941176 ,0.43529412 ,1.        ]





# Create subplots
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 12))

#font size
fontSize = None

# Plot Mean Absolute Error
axes[0,0].bar(x, mae_values, width=bar_width, label='Base', color=colorA)
axes[0,0].bar(x + bar_width + bar_space, mae_values3, width=bar_width, label='HP', color=colorF)
axes[0,0].bar(x + 2 * (bar_width + bar_space), mae_values2, width=bar_width, label='Base + Poly F.E.', color=colorC)
axes[0,0].bar(x + 3 * (bar_width + bar_space), mae_values4, width=bar_width, label='HP + Poly F.E.', color=colorE)
axes[0,0].tick_params(axis='x', which='both', bottom=False, top=False)  # Disable x-axis ticks and labels
axes[0,0].set_xticklabels([])  # Remove x-axis labels
axes[0,0].set_ylim([0, 0.06])
# axes[0].set_yscale('log')
axes[0,0].set_title('Mean Absolute Error (MAE)', x = 1.1, fontsize= fontSize)
# axes[0].set_ylabel('MAE')


# Plot Root Mean Squared Error
axes[1,0].bar(x, rmse_values, width=bar_width, label='Base', color=colorA)
axes[1,0].bar(x + bar_width + bar_space, rmse_values3, width=bar_width, label='HP', color=colorF)
axes[1,0].bar(x + 2 * (bar_width + bar_space), rmse_values2, width=bar_width, label='Base + Poly F.E.', color=colorC)
axes[1,0].bar(x + 3 * (bar_width + bar_space), rmse_values4, width=bar_width, label='HP + Poly F.E.', color=colorE)
axes[1,0].tick_params(axis='x', which='both', bottom=False, top=False)  # Disable x-axis ticks and labels
axes[1,0].set_xticklabels([])  # Remove x-axis labels
# axes[1].set_yscale('log')
# axes[1].set_ylim([0,0.1])
axes[1,0].set_title('Root Mean Squared Error (RMSE)', x = 1.1, fontsize= fontSize)
# axes[1].set_ylabel('RMSE')
axes[1,0].set_ylim([0, 0.1])


# Plot Relative Root Squared Error
axes[2,0].bar(x, rrse_values, width=bar_width, label='Base', color=colorA)
axes[2,0].bar(x + bar_width, rrse_values3, width=bar_width, label='HP', color=colorF)
axes[2,0].bar(x + 2 * bar_width, rrse_values2, width=bar_width, label='Base + Poly F.E.', color=colorC)
axes[2,0].bar(x + 3 * bar_width, rrse_values4, width=bar_width, label='HP + Poly F.E.', color=colorE)
axes[2,0].tick_params(axis='x', which='both', bottom=False, top=False)  # Disable x-axis ticks and labels
axes[2,0].set_xticklabels([])  # Remove x-axis labels
# axes[1].set_yscale('log')
axes[2,0].set_ylim([0, 1])
axes[2,0].set_title('Relative Root Squared Error (RRSE)', x = 1.1, fontsize= fontSize)
# axes[2,0].set_ylabel('Error Values', fontsize= fontSize)
# axes[2].set_ylabel('RRSE')
# axes[2].set_ylim([0.25, 0.35])


# Plot Relative Absolute Error
axes[3,0].bar(x, rae_values, width=bar_width, label='Base', color=colorA)
axes[3,0].bar(x + bar_width, rae_values3, width=bar_width, label='HP', color=colorF)
axes[3,0].bar(x + 2 * bar_width, rae_values2, width=bar_width, label='Base + Poly F.E.', color=colorC)
axes[3,0].bar(x + 3 * bar_width, rae_values4, width=bar_width, label='HP + Poly F.E.', color=colorE)
axes[3,0].tick_params(axis='x', which='both', bottom=False, top=False)  # Disable x-axis ticks and labels
axes[3,0].set_xticklabels([])  # Remove x-axis labels
axes[3,0].set_ylim([0, 1])
axes[3,0].set_title('Relative Absolute Error (RAE)', x = 1.1, fontsize= fontSize)
# axes[3].set_ylabel('RAE')
# axes[3].set_ylim([0.12, 0.22])


axes[4,0].bar(x, correlation_values, width=bar_width, label='Base', color=colorA)
axes[4,0].bar(x + bar_width, correlation_values3, width=bar_width, label='HP', color=colorF)
axes[4,0].bar(x + 2 * bar_width, correlation_values2, width=bar_width, label='Base + Poly F.E.', color=colorC)
axes[4,0].bar(x + 3 * bar_width, correlation_values4, width=bar_width, label='HP + Poly F.E.', color=colorE)
axes[4,0].set_ylim([0, 1])
axes[4,0].set_title('Correlation Coefficient', x = 1.1, fontsize= fontSize)


# Add legend to the last subplot
axes[0,0].legend(loc='upper center', bbox_to_anchor=(1.1, 1.4), ncol=6, fontsize= fontSize)
axes[4,0].set_xticks(x, models, rotation=45, ha='right', fontsize= fontSize)



var = 'Width'
df = pd.read_csv(f'LPBF_ML_Model_Evaluation_BaseModel_{var}.csv')
df2 = pd.read_csv(f'Poly6{var}_base.csv')

# HP
df3 = pd.read_csv(f'LPBF_ML_Model_Evaluation_HpTuning_{var}.csv')
df4 = pd.read_csv(f'LPBF_ML_Model_Evaluation_Poly6_{var}.csv')



# Read data from CSV file
# df = pd.read_csv('width_new_model_comparison_results_20231214192952.csv')  # Replace with the actual path to your CSV file

# Extract model names and error values
models = df['Model'].tolist()

# BASE
mae_values = df['MAE'].tolist()
rmse_values = df['RMSE'].tolist()
rrse_values = df['RRSE'].tolist()
rae_values = df['RAE'].tolist()
correlation_values = df['Correlation Coefficient'].tolist()
# correlation_values = df['R-squared'].tolist()

# Poly Base
mae_values2 = df2['MAE'].tolist()
rmse_values2 = df2['RMSE'].tolist()
rrse_values2 = df2['RRSE'].tolist()
rae_values2 = df2['RAE'].tolist()
# correlation_values2 = df2['Correlation Coefficient'].tolist()
correlation_values2 = df2['R-squared'].tolist()

# HP BASE
mae_values3 = df3['MAE'].tolist()
rmse_values3 = df3['RMSE'].tolist()
rrse_values3 = df3['RRSE'].tolist()
rae_values3 = df3['RAE'].tolist()
correlation_values3 = df3['Correlation Coefficient'].tolist()

# HP POLY

mae_values4 = df4['MAE'].tolist()
rmse_values4 = df4['RMSE'].tolist()
rrse_values4 = df4['RRSE'].tolist()
rae_values4 = df4['RAE'].tolist()
correlation_values4 = df4['Correlation Coefficient'].tolist()

models_n = ['GP', 'LR', 'PR', 'SVR', 'KNN', 'MLP', 'RF', 'GB', 'AB', 'B', 'ET']


# Plot Mean Absolute Error
axes[0,1].bar(x, mae_values, width=bar_width, label='Base', color=colorA)
axes[0,1].bar(x + bar_width + bar_space, mae_values3, width=bar_width, label='HP', color=colorF)
axes[0,1].bar(x + 2 * (bar_width + bar_space), mae_values2, width=bar_width, label='Base + Poly F.E.', color=colorC)
axes[0,1].bar(x + 3 * (bar_width + bar_space), mae_values4, width=bar_width, label='HP + Poly F.E.', color=colorE)
axes[0,1].tick_params(axis='x', which='both', bottom=False, top=False)  # Disable x-axis ticks and labels
axes[0,1].set_xticklabels([])  # Remove x-axis labels
axes[0,1].set_ylim([0, 0.06])
# axes[0].set_yscale('log')
# axes[0,1].set_title('Mean Absolute Error (MAE)', fontsize= fontSize)
# axes[0].set_ylabel('MAE')


# Plot Root Mean Squared Error
axes[1,1].bar(x, rmse_values, width=bar_width, label='Base', color=colorA)
axes[1,1].bar(x + bar_width + bar_space, rmse_values3, width=bar_width, label='HP', color=colorF)
axes[1,1].bar(x + 2 * (bar_width + bar_space), rmse_values2, width=bar_width, label='Base + Poly F.E.', color=colorC)
axes[1,1].bar(x + 3 * (bar_width + bar_space), rmse_values4, width=bar_width, label='HP + Poly F.E.', color=colorE)
axes[1,1].tick_params(axis='x', which='both', bottom=False, top=False)  # Disable x-axis ticks and labels
axes[1,1].set_xticklabels([])  # Remove x-axis labels
# axes[1].set_yscale('log')
# axes[1].set_ylim([0,0.1])
# axes[1,1].set_title('Root Mean Squared Error (RMSE)', fontsize= fontSize)
# axes[1].set_ylabel('RMSE')
axes[1,1].set_ylim([0, 0.1])


# Plot Relative Root Squared Error
axes[2,1].bar(x, rrse_values, width=bar_width, label='Base', color=colorA)
axes[2,1].bar(x + bar_width, rrse_values3, width=bar_width, label='HP', color=colorF)
axes[2,1].bar(x + 2 * bar_width, rrse_values2, width=bar_width, label='Base + Poly F.E.', color=colorC)
axes[2,1].bar(x + 3 * bar_width, rrse_values4, width=bar_width, label='HP + Poly F.E.', color=colorE)
axes[2,1].tick_params(axis='x', which='both', bottom=False, top=False)  # Disable x-axis ticks and labels
axes[2,1].set_xticklabels([])  # Remove x-axis labels
# axes[1].set_yscale('log')
axes[2,1].set_ylim([0, .6])
# axes[2,1].set_title('Relative Root Squared Error (RRSE)', fontsize= fontSize)
# axes[2,1].set_ylabel('Error Values', fontsize= fontSize)
# axes[2].set_ylabel('RRSE')
# axes[2].set_ylim([0.25, 0.35])


# Plot Relative Absolute Error
axes[3,1].bar(x, rae_values, width=bar_width, label='Base', color=colorA)
axes[3,1].bar(x + bar_width, rae_values3, width=bar_width, label='HP', color=colorF)
axes[3,1].bar(x + 2 * bar_width, rae_values2, width=bar_width, label='Base + Poly F.E.', color=colorC)
axes[3,1].bar(x + 3 * bar_width, rae_values4, width=bar_width, label='HP + Poly F.E.', color=colorE)
axes[3,1].tick_params(axis='x', which='both', bottom=False, top=False)  # Disable x-axis ticks and labels
axes[3,1].set_xticklabels([])  # Remove x-axis labels
axes[3,1].set_ylim([0, .6])
# axes[3,1].set_title('Relative Absolute Error (RAE)', fontsize= fontSize)
# axes[3].set_ylabel('RAE')
# axes[3].set_ylim([0.12, 0.22])



axes[4,1].bar(x, correlation_values, width=bar_width, label='Base', color=colorA)
axes[4,1].bar(x + bar_width, correlation_values3, width=bar_width, label='HP', color=colorF)
axes[4,1].bar(x + 2 * bar_width, correlation_values2, width=bar_width, label='Base + Poly F.E.', color=colorC)
axes[4,1].bar(x + 3 * bar_width, correlation_values4, width=bar_width, label='HP + Poly F.E.', color=colorE)
axes[4,1].tick_params(axis='x', which='both', bottom=False, top=False)  # Disable x-axis ticks and labels
axes[4,1].set_xticklabels([])  # Remove x-axis labels
axes[4,1].set_ylim([0, 1])
# axes[4,1].set_title('Correlation Coefficient', fontsize= fontSize)
axes[4,1].set_xticks(x, models, rotation=45, ha='right', fontsize= fontSize)







# #Adjust layout for better spacing
# plt.tight_layout()


# Save the plot as a PNG file
plt.savefig(f'LPBF_Plot_barChart_fig8.png', bbox_inches='tight')

# # Show the plot
# plt.show()