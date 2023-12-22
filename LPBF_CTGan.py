import pandas as pd
from ctgan import *
import ctgan
from ctgan import CTGAN
# from CTGAN import*
# from ctgan import CTGANSynthesizer


# Load your real dataset
real_data = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Original.csv')
print(real_data.columns)

categorical_features = ['Laser Power [W]', 'Scanning Speed [mm/s]', 'Layer Thickness [um]',
       'Spot Size [um]', 'Porosity [%]', 'Max Melt Pool Width [um]', 'Max Melt Pool Depth [um]']

# ctgan = CTGAN(verbose=True)
# ctgan.fit(real_data, categorical_features, epochs = 400)
#
# samples = ctgan.sample(1000)
#
# print(samples.head())
#
#
# samples.to_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Synthetic_1000.csv', index=False)
synthetic_data = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Synthetic_1000.csv')

# fig = report.get_visualization(property_name = 'Column Shapes')
# fig.show()
# report.get_score()

from table_evaluator import load_data, TableEvaluator
table_evaluator = TableEvaluator(real_data, synthetic_data)
table_evaluator.visual_evaluation()

from sdmetrics.reports.single_table import QualityReport

metadata = {
       "columns": {
              "Laser Power [W]": {
                  "sdtype": "numerical",
                  "compute_representation": "Float"
                       },
              "Scanning Speed [mm/s]": {
                  "sdtype": "numerical",
                  "compute_representation": "Float"
              },
              "Layer Thickness [um]": {
                  "sdtype": "numerical",
                  "compute_representation": "Float"
              },
              "Spot Size [um]": {
                  "sdtype": "numerical",
                  "compute_representation": "Float"
              },
              "Porosity [%]": {
                  "sdtype": "numerical",
                  "compute_representation": "Float"
              },
              "Max Melt Pool Width [um]": {
                  "sdtype": "numerical",
                  "compute_representation": "Float"
              },
              "Max Melt Pool Depth [um]": {
                  "sdtype": "numerical",
                  "compute_representation": "Float"
              },
               },
}

report = QualityReport()

report.generate(real_data, synthetic_data, metadata)

report.get_score()


print(report.get_details(property_name='Column Shapes'))