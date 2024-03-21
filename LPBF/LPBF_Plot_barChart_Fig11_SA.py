import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import get_cmap
import numpy as np
import textwrap


params = ['Laser\nPower\n[W]','Scanning\nSpeed\n[mm/s]','Layer\nThickness\n[um]','Spot\nSize\n[um]','Porosity\n[%]']
# values_width = [pow(10,0.820867759), pow(10,0.279461045), pow(10,0.001164634), pow(10,0.008927962), pow(10,0.016793913)]
# values_depth = [pow(10,0.70479538), pow(10,0.404152123), pow(10,0.000530379), pow(10,0.001178912), pow(10,0.001229683)]

values_width = [0.820867759, 0.279461045, 0.001164634, 0.008927962, 0.016793913]
values_depth = [0.70479538,0.404152123,0.000530379,0.001178912,0.001229683]

# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=2,tight_layout=True,  figsize=(8, 4))
#sharey=True,
#adjust the horizontal space between two plots
plt.subplots_adjust(wspace=0.3)



bar_labels = ['red', 'blue', 'green', 'orange', 'gray']
bar_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange','tab:gray']

axes[0].bar(params, values_width, label=bar_labels, color=bar_colors)
axes[0].set_ylabel('Sensitivity Value', labelpad = 2)
axes[0].set_title('Melt-Pool Width')
axes[0].set_yscale('log')  # Setting y-scale to logarithmic (base 10)


axes[1].bar(params, values_depth, label=bar_labels, color=bar_colors)
# axes[1].set_ylabel('Sensitivity Value', labelpad = 2)
axes[1].set_title('Melt-Pool Depth')
axes[1].set_yscale('log')  # Setting y-scale to logarithmic (base 10)



# Save the plot as a PNG file
plt.savefig(f'LPBF_Plot_barChart_Fig11_SA.png', bbox_inches='tight', dpi = 1200)

plt.show()