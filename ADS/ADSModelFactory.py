import os
from ADS.ADSModelsVersions.ADSModelSimple import ADSModelSimple
from ADS.ADSModelsVersions.ADSModelRandom import ADSModelRandom
from ADS.ADSModelsVersions.ADSModelTunning import ADSModelTunning


class ADSModelFactory:

    @staticmethod
    def getADSModelVersion(iter_retraining=None):
        if os.environ.get('DIGITAL_MODEL_RETRAINING_TEST_SIZE') == "SIMPLE":
            return ADSModelSimple(iter_retraining)
        elif os.environ.get('DIGITAL_MODEL_RETRAINING_TEST_SIZE') == "RANDOM":
            return ADSModelRandom(iter_retraining)
        elif os.environ.get('DIGITAL_MODEL_RETRAINING_TEST_SIZE') == "TUNNING":
            return ADSModelTunning(iter_retraining)
        else:
            return ADSModelRandom(iter_retraining)
