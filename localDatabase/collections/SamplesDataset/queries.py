from localDatabase.collections.client import db
import pymongo
import datetime
import os

collection = db.SamplesDataset

def setDatasetsPath(config):
    config["timestamp"] = datetime.datetime.now().timestamp()
    result = collection.insert_one(config)
    return result

def getPathDatasetCsv():
    path = os.environ.get('DIGITAL_MODEL_DATASET_PATH')
    if not path or path == "":
        result = collection.find_one(sort=[("timestamp", pymongo.DESCENDING)])
        path = result.get("pathDatasetCsv")
    return path

def getPathDatasetReviewedCsv():
    path = os.environ.get('DIGITAL_MODEL_DATASET_REVIEWED_PATH')
    if not path or path == "":
        result = collection.find_one(sort=[("timestamp", pymongo.DESCENDING)])
        path = result.get("pathDatasetReviewedCsv")
    return path

def getPathDatasetHighAnomaliesCsv():
    path = os.environ.get('DIGITAL_MODEL_DATASET_HIGH_ANOMALIES_PATH')
    if not path or path == "":
        result = collection.find_one(sort=[("timestamp", pymongo.DESCENDING)])
        path = result.get("pathDatasetHighAnomaliesCsv")
    return path

def getPathResizedImage():
    result = collection.find_one(sort=[("timestamp", pymongo.DESCENDING)])
    path = result.get("pathResizedImage")
    return path

def getPathObjectsImage():
    result = collection.find_one(sort=[("timestamp", pymongo.DESCENDING)])
    path = result.get("pathObjectsImage")
    return path

def getPathSurfacesImage():
    result = collection.find_one(sort=[("timestamp", pymongo.DESCENDING)])
    path = result.get("pathSurfacesImage")
    return path