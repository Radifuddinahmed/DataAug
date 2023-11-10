import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(font_scale=1.2)
plt.figure(figsize=(30, 30))
raw_data = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Book2.csv',encoding='latin1')
sns.pairplot(raw_data)

plt.savefig('myfig_LPBF_Depth_V2.png', dpi=1200, bbox_inches='tight')


