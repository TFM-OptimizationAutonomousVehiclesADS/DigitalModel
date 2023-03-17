from ADS.ADSModel import ADSModel
import logging
import time
import random

SLEEP_TIME = 60*2

if __name__ == "__main__":
    logging.info("** RETRAINING TASK: Iniciando Modelo de Detección de Anomalías....")
    adsModel = ADSModel()
    size_split = 5000
    epochs = 10
    while True:
        try:
            logging.info("** RETRAINING TASK: Comenzando Reentrenamiento....")
            adsModel.retrain_model(random=True, size_split=size_split, epochs=epochs, tunning=False, model_by_best_epoch=True)
            logging.info("** RETRAINING TASK: FIN REENTRENAMIENTO")
            size_split += 100
            epochs = random.randint(10, 100) #inclucive

        except Exception as e:
            logging.exception("Error en ADS: " + str(e))

        time.sleep(SLEEP_TIME)