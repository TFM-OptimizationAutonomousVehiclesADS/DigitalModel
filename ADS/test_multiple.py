from ADS.ADSModelFactory import ADSModelFactory
from ADS.ADSModelsVersions.ADSModelMultiple import ADSModelMultiple
import logging
import time
import random
import os
import json

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
    best_epoch = int(os.environ.get('DIGITAL_MODEL_RETRAINING_BEST_EPOCH', 0))
    retrain_weights = int(os.environ.get('DIGITAL_MODEL_RETRAINING_RETRAIN_WEIGHTS', 0))
    random_samples = int(os.environ.get('DIGITAL_MODEL_RETRAINING_RANDOM_SAMPLES', 1))
    iter_retraining = 1

    with open("/opt/DigitalModel/ADS/model_configs_test.json", "r") as f:
        model_configs = json.load(f)

    print(json.dumps(model_configs))
    os.environ.setdefault("DIGITAL_MODEL_COMBINE_MODEL_CONFIGS", json.dumps(model_configs))

    adsModel = ADSModelMultiple(iter_retraining=iter_retraining)

    retraining = False
    if adsModel.models:
        print("TIENE MODELOS")
        # print(adsModel.get_actual_evaluation_model())
        retraining = True

    iter_retraining = iter_retraining + 1
    try:
        size_split = random.randint(min_size_split, max_size_split)
        epochs = random.randint(min_epochs, max_epochs)

        logging.info("** RETRAINING TASK: Comenzando Reentrenamiento....")
        best_retrain_model = adsModel.retrain_model(retraining=retraining, test_size=test_size, random=random_samples, retrainWeights=False, size_split=size_split, epochs=10, tunning=False, model_by_best_epoch=False)
        logging.info("** RETRAINING TASK: FIN REENTRENAMIENTO")

        if best_retrain_model:
            logging.info("** RETRAINING TASK: BEST RETRAIN MODEL FOUND")
            # TODO SEND NOTIFICATION TO API CENTRAL SYSTEM

    except Exception as e:
        logging.exception("Error en ADS: " + str(e))
