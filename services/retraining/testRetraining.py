
from ADS.ADSModel import ADSModel
import pandas as pd
from localDatabase.collections.SamplesDataset.queries import getPathDatasetCsv

adsModel = ADSModel()
adsModel.train_model(random=True, size_split=100)