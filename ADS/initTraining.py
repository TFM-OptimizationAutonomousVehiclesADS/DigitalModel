
from ADS.ADSModelsVersions.ADSModelSimple import ADSModelSimple
import logging
import os

if __name__ == "__main__":
    test_size = float(os.environ.get('DIGITAL_MODEL_RETRAINING_TEST_SIZE', 0.25))
    min_size_split = int(os.environ.get('DIGITAL_MODEL_RETRAINING_MIN_SPLIT', 2000))
    min_epochs = int(os.environ.get('DIGITAL_MODEL_RETRAINING_MIN_EPOCHS', 10))
    best_epoch = int(os.environ.get('DIGITAL_MODEL_RETRAINING_BEST_EPOCH', 1))
    random_samples = int(os.environ.get('DIGITAL_MODEL_RETRAINING_RANDOM_SAMPLES', 1))

    logging.info("** INIT TRAINING: Iniciando Modelo de Detección de Anomalías....")
    adsModel = ADSModelSimple()
    if adsModel.model is None:
        logging.info("** INIT TRAINING: Comenzando entrenamiento....")
        adsModel.retrain_model(random=random_samples, size_split=min_size_split, test_size=test_size, epochs=min_epochs, model_by_best_epoch=best_epoch, retraining=False)
        logging.info("** INIT TRAINING: FIN ENRENAMIENTO")