
from ADS.ADSModel import ADSModel
import pandas as pd
from localDatabase.collections.SamplesDataset.queries import getPathDatasetCsv

adsModel = ADSModel()

pathDatasetCsv = getPathDatasetCsv()
dataset = pd.read_csv(pathDatasetCsv)

samples = dataset.sample(n=1)
sampleRow = samples.iloc[0]
sampleRowJson = sampleRow.to_dict()

y_pred = adsModel.predict_sample(sampleRowJson)
print(y_pred)

anomaly = adsModel.is_anomaly(y_pred)
print(anomaly)