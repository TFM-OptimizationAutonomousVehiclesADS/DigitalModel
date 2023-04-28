from localDatabase.queues.queueSamples import QueueSamples
from localDatabase.collections.DetectedAnomalies.queries import addAnomalySample
from localDatabase.collections.PredictionLogs.queries import addPredictionLogSample
from ADS.ADSModel import ADSModel
import datetime
import logging

queueSamples = QueueSamples()

if __name__ == "__main__":
    # adsModel = ADSModel()
    while True:
        try:
            logging.info("WAITING NEXT SAMPLE...")
            # Wait next sample
            sample = queueSamples.waitSampleFromQueue()
            logging.info("Next sample:")
            logging.info(sample)

            adsModel = ADSModel()

            y = sample["anomaly"]
            y_pred = adsModel.predict_sample(sample)
            logging.info("REAL ANOMALY: " + str(y))
            logging.info("PREDICTION: " + str(y_pred))

            sample["prediction"] = y_pred
            sample["timestamp"] = datetime.datetime.now()

            img_resized, obj_resized, surf_resized = adsModel.get_resized_images_base64(sample)
            sample["image_resized_base64"] = img_resized
            sample["object_resized_base64"] = obj_resized
            sample["surface_resized_base64"] = surf_resized

            is_predicted_anomaly = adsModel.is_anomaly(y_pred)
            if is_predicted_anomaly:
                logging.info("ES UNA ANOMALIA, GUARDANDO EN BASE DE DATOS")
                addAnomalySample(sample)
            addPredictionLogSample(sample)

        except Exception as e:
            logging.exception("Error en ADS: " + str(e))