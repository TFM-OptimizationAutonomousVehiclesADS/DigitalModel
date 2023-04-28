from ADS.ADSModel import ADSModel
import logging
import time
import random
import os

SLEEP_TIME = 60*2

if __name__ == "__main__":
    logging.info("** RETRAINING TASK: Iniciando Modelo de Detección de Anomalías....")

    is_real_system = int(os.environ.get('IS_REAL_SYSTEM', 0))
    if is_real_system:
        exit(1)

    test_size = float(os.environ.get('DIGITAL_MODEL_RETRAINING_TEST_SIZE', 0.25))
    min_size_split = int(os.environ.get('DIGITAL_MODEL_RETRAINING_MIN_SPLIT', 2000))
    max_size_split = int(os.environ.get('DIGITAL_MODEL_RETRAINING_MAX_SPLIT', 5000))
    min_epochs = int(os.environ.get('DIGITAL_MODEL_RETRAINING_MIN_EPOCHS', 10))
    max_epochs = int(os.environ.get('DIGITAL_MODEL_RETRAINING_MAX_EPOCHS', 100))
    tunning = int(os.environ.get('DIGITAL_MODEL_RETRAINING_TUNNING', 0))
    best_epoch = int(os.environ.get('DIGITAL_MODEL_RETRAINING_BEST_EPOCH', 1))
    retrain_weights = int(os.environ.get('DIGITAL_MODEL_RETRAINING_RETRAIN_WEIGHTS', 1))
    random_samples = int(os.environ.get('DIGITAL_MODEL_RETRAINING_RANDOM_SAMPLES', 1))
    iter_retraining = 1

    while True:
        adsModel = ADSModel(iter_retraining=iter_retraining)
        iter_retraining = iter_retraining + 1
        try:
            size_split = random.randint(min_size_split, max_size_split)
            epochs = random.randint(min_epochs, max_epochs)

            logging.info("** RETRAINING TASK: Comenzando Reentrenamiento....")
            adsModel.retrain_model(test_size=test_size, random=random_samples, retrainWeights=retrain_weights, size_split=size_split, epochs=epochs, tunning=tunning, model_by_best_epoch=best_epoch)
            logging.info("** RETRAINING TASK: FIN REENTRENAMIENTO")

        except Exception as e:
            logging.exception("Error en ADS: " + str(e))

        time.sleep(SLEEP_TIME)