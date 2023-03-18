
from ADS.ADSModel import ADSModel
import pandas as pd
from localDatabase.collections.SamplesDataset.queries import getPathDatasetCsv

adsModel = ADSModel()

pathDatasetCsv = getPathDatasetCsv()
dataset = pd.read_csv(pathDatasetCsv)

samples = dataset.sample(n=1)
sampleRow = samples.iloc[0]
sampleRowJson = sampleRow.to_dict()

img_resized, obj_resized, surf_resized = adsModel.get_resized_images_base64(sampleRowJson)
print(img_resized)

# y_pred = adsModel.predict_sample(sampleRowJson)
# print(y_pred)
#
# anomaly = adsModel.is_anomaly(y_pred)
# print(anomaly)