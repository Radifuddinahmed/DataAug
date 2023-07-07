import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(font_scale=1.2)
plt.figure(figsize=(30, 30))
raw_data = pd.read_csv('FSW_Dataset_v4_test.csv',encoding='latin1')
sns.pairplot(raw_data)

plt.savefig('myfig.png', dpi=300, bbox_inches='tight')


