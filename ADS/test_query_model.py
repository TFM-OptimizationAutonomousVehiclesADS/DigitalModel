from ADS.ADSModelFactory import ADSModelFactory
import logging
import time
import random
import os

if __name__ == "__main__":
    adsModel = ADSModelFactory.getADSModelVersion()
    model_json = adsModel.get_actual_model_json()
    print(model_json)