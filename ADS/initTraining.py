
from ADS.ADSModel import ADSModel
import pandas as pd
from localDatabase.collections.SamplesDataset.queries import getPathDatasetCsv
import logging

if __name__ == "__main__":
    logging.info("** INIT TRAINING: Iniciando Modelo de Detección de Anomalías....")
    adsModel = ADSModel()
    logging.info("** INIT TRAINING: Comenzando entrenamiento....")
    adsModel.train_new_model(random=True, size_split=2000)
    logging.info("** INIT TRAINING: FIN ENRENAMIENTO")