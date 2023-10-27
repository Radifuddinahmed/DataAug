import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

well_data = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\LPBF_Dataset_Combined_v2_python_Width+Depth_no_zero_por_data.csv',encoding='latin1')


corr = well_data.corr()
sns.set(font_scale=3)
sns.set_style({'font.family': 'Times New Roman'})
# # Set the horizontal alignment to "wrap"
# corr.set(horizontal_alignment="wrap")


plt.figure(figsize=(30, 20))
#plt.figure(figsize=(19, 6))
heatmap = sns.heatmap(well_data.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG' , fmt='.3f', square=True) #.1f = one decimal place
heatmap.set_title('  ', fontdict={'fontsize':6}, pad=22)
plt.xticks(rotation=90)



plt.savefig('heatmap_LPBF.png', dpi=300, bbox_inches='tight')

