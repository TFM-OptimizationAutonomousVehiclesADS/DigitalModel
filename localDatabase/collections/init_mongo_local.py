from localDatabase.collections.ApiAccessConfiguration.queries import setActualIpAndPort
from localDatabase.collections.MLModelConfiguration.queries import setModelConfig
from localDatabase.collections.SamplesDataset.queries import setDatasetsPath
import logging

logging.info("INICIALIZANDO CONFIGURACION DE MONGODB...")

rootPath = "/opt/DigitalModel"
# rootPath = "/opt/tfm/OptimizationAutonomousVehiclesADS/DigitalModel"

setActualIpAndPort("127.0.0.1", 8001)

setModelConfig({
  "imageModelPath": rootPath + "/models/model_image.png",
  "evaluationModelPath": rootPath + "/models/evaluation_model.json",
  "modelPath": rootPath + "/models/model",
  "compileOptimizer": {
    "optimizer": "adam",
    "loss": "binary_crossentropy"
  },
  "floatFeatures": [
    "channel_camera",
    "speed",
    "rotation_rate_z"
  ],
  "imagesFeatures": [
    "filename_resized_image",
    "filename_objects_image",
    "filename_surfaces_image"
  ],
  "sizeImageHeight": {
    "$numberLong": "45"
  },
  "sizeImageWidth": {
    "$numberLong": "80"
  },
  "features": [
    "channel_camera",
    "speed",
    "rotation_rate_z",
    "filename_resized_image",
    "filename_objects_image",
    "filename_surfaces_image"
  ],
  "threshold": 0.5
})

setDatasetsPath({
  "pathDatasetCsv": rootPath + "/datasets/dataset_all_no_missclassification.csv",
  "pathDatasetReviewedCsv": rootPath + "/datasets/dataset_reviewed.csv",
  "pathDatasetHighAnomaliesCsv": rootPath + "/datasets/dataset_high_anomalies.csv",
  "pathResizedImage": rootPath + "/datasets/resized_images",
  "pathObjectsImage": rootPath + "/datasets/objects_images",
  "pathSurfacesImage": rootPath + "/datasets/surfaces_images"
})

logging.info("INICIALIZADO CONFIGURACION DE MONGODB")