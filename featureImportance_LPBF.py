from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
import pandas  as pd


# Load the dataset as an example
df = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\LPBF_Dataset_Combined_v1_python.csv')
X = df[['Laser Power [W]', 'Scanning Speed [mm/s]', 'Layer Thickness [um]', 'Spot Size [um]', 'Porosity [%]']].values
y = df['Max Melt Pool Width [um]'].values

model = LinearRegression()
model.fit(X,y)

importance = model.coef_

for i,v in enumerate(importance):
    print("Feature: %0d, Score: %.5f" %(i,v))

pyplot.bar([x for x in range (len(importance))], importance)
pyplot.show()

