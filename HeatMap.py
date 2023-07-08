import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

well_data = pd.read_csv('FSW_Dataset2.csv',encoding='latin1')

corr = well_data.corr()
sns.set(font_scale=3)
sns.set_style({'font.family': 'Times New Roman'})


plt.figure(figsize=(12, 12))
#plt.figure(figsize=(19, 6))
heatmap = sns.heatmap(well_data.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('  ', fontdict={'fontsize':6}, pad=22)

plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
