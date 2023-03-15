import time
from localDatabase.collections.SamplesDataset.queries import getPathDatasetCsv
from localDatabase.collections.ApiAccessConfiguration.queries import getActualIpAndPort
import pandas as pd
import requests
import logging

# export PYTHONPATH=/mnt/d/Escritorio2.0/UMA/MasterInformaticaUMA/TFM/OptimizationAutonomousVehiclesADS/DigitalModel

SLEEP_TIME = 60

if __name__ == "__main__":
    while True:
        try:
            pathDatasetCsv = getPathDatasetCsv()
            dataset = pd.read_csv(pathDatasetCsv)

            samples = dataset.sample(n=1)
            sampleRow = samples.iloc[0]
            sampleRowJson = sampleRow.to_dict()
            logging.info(sampleRowJson)

            ipApi, portApi = getActualIpAndPort()
            headers = {
                "Content-Type": "application/json",
                "accept": "application/json"
            }
            dataPost = {'sample': sampleRowJson}
            r = requests.post("http://" + str(ipApi) + ":" + str(portApi) + "/listen_sample", json=sampleRowJson, headers=headers)
            logging.info("RESPONE: ")
            logging.info(r.status_code)
            logging.info(r.json())

        except Exception as e:
            logging.exception(e)

        time.sleep(SLEEP_TIME)