import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



well_data = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\LPBF_Dataset_Combined_v2_python_Width+Depth_no_zero_por_data.csv',encoding='latin1')

# Create a box plot using Seaborn
BoxPlot = sns.boxplot(data=well_data)
plt.xticks(rotation=90)

plt.savefig('BoxPlot_LPBF.png', dpi=100, bbox_inches='tight')