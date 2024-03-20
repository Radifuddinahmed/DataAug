import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(font_scale=1.2) #original 1.2
plt.figure(figsize=(30, 30)) #original 1.2
raw_data = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Normalized.csv',encoding='latin1')
sns.pairplot(raw_data, corner=True)

plt.savefig('LPBF_Plot_scatter_fig3.png', dpi=600, bbox_inches='tight')


