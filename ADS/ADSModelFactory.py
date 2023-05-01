import os
from ADS.ADSModelsVersions.ADSModelSimple import ADSModelSimple
from ADS.ADSModelsVersions.ADSModelMultiple import ADSModelMultiple
from ADS.ADSModelsVersions.ADSModelRandom import ADSModelRandom
from ADS.ADSModelsVersions.ADSModelTunning import ADSModelTunning
from ADS.ADSModelsVersions.ADSModelCombinated import ADSModelCombinated


class ADSModelFactory:

    @staticmethod
    def getADSModelVersion(iter_retraining=None):
        if os.environ.get('DIGITAL_MODEL_VERSION') == "SIMPLE":
            print("********** DIGITAL MODEL VERSION: SIMPLE *********")
            return ADSModelSimple(iter_retraining)
        elif os.environ.get('DIGITAL_MODEL_VERSION') == "RANDOM":
            print("********** DIGITAL MODEL VERSION: RANDOM *********")
            return ADSModelRandom(iter_retraining)
        elif os.environ.get('DIGITAL_MODEL_VERSION') == "TUNNING":
            print("********** DIGITAL MODEL VERSION: TUNNING *********")
            return ADSModelTunning(iter_retraining)
        elif os.environ.get('DIGITAL_MODEL_VERSION') == "MULTIPLE":
            print("********** DIGITAL MODEL VERSION: MULTIPLE *********")
            return ADSModelMultiple(iter_retraining)
        elif os.environ.get('DIGITAL_MODEL_VERSION') == "COMBINATED":
            print("********** DIGITAL MODEL VERSION: COMBINATED *********")
            return ADSModelCombinated(iter_retraining)
        else:
            return ADSModelRandom(iter_retraining)
