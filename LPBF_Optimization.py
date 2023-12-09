import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
# Load the dataset as an example
df = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Original_Width.csv')
X = df[['Laser Power [W]', 'Scanning Speed [mm/s]', 'Layer Thickness [um]', 'Spot Size [um]', 'Porosity [%]']].values
y = df['Max Melt Pool Width [um]'].values

# Create an Extra Trees Regressor
ET_regressor = ExtraTreesRegressor(max_depth=10,min_samples_split=5,n_estimators=10, random_state=42)

ET_regressor.fit(X, y)


c1 = []
c2 = []
c3 = []
c4 = []
c5 = []
c6 = []
opt_data = {
            'Laser Power [W]': c1,
            'Scanning Speed [mm/s]': c2,
            'Layer Thickness [um]': c3,
            'Spot Size [um]': c4,
            'Porosity [%]': c5,
            'Max Melt Pool Width [um]': c6
            }
opt = pd.DataFrame(opt_data)

for laserPower in np.arange(50,520,10):
    for scanningSpeed in np.arange(0, 3200, 100):
        for layerThickness in np.arange(30, 250, 10):
            spotSize = 100
            porosity = 50
            width = (ET_regressor.predict([[laserPower,scanningSpeed,layerThickness,spotSize,porosity]]))
            opt = pd.concat([opt, pd.DataFrame.from_records([{
                                                            'Laser Power [W]': laserPower,
                                                            'Scanning Speed [mm/s]': scanningSpeed,
                                                            'Layer Thickness [um]': spotSize,
                                                            'Spot Size [um]': spotSize,
                                                            'Porosity [%]': porosity,
                                                            'Max Melt Pool Width [um]': width.item()
                                                            }])])


print(width.item())
opt.to_csv('LPBF_optimization_width.csv',mode='w', index=False)