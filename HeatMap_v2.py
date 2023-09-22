import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv('Data Sheet Revised_CSV.csv')

# Create a heatmap of the DataFrame
sns.heatmap(df)

# Create a correlation matrix of the DataFrame
corr = df.corr()

# Use the Seaborn `heatmap()` function to plot the correlation matrix
sns.heatmap(corr)

# Display the heatmap
plt.show()