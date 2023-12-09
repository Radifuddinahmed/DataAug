import pandas as pd
from ctgan import *
import ctgan
from ctgan import CTGAN
# from CTGAN import*
# from ctgan import CTGANSynthesizer


# Load your real dataset
real_data = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Original_Width.csv')
print(real_data.columns)

categorical_features = ['Laser Power [W]', 'Scanning Speed [mm/s]', 'Layer Thickness [um]',
       'Spot Size [um]', 'Porosity [%]', 'Max Melt Pool Width [um]']

ctgan = CTGAN(verbose=True)
ctgan.fit(real_data, categorical_features, epochs = 300)

samples = ctgan.sample(1000)

print(samples.head())


samples.to_csv('LPBF_synthetic_data.csv', index=False)

