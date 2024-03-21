import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



well_data = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Book2.csv',encoding='latin1')

# Create a box plot using Seaborn
plt.figure(figsize=(5, 10))
BoxPlot = sns.boxplot(data=well_data)
plt.xticks(rotation=0)

plt.savefig('BoxPlot_LPBF_porosity.png', dpi=300, bbox_inches='tight')