from localDatabase.collections.client import db
import pymongo
import datetime

collection = db.SamplesDataset

def setDatasetsPath(config):
    config["timestamp"] = datetime.datetime.now().timestamp()
    result = collection.insert_one(config)
    return result

def getPathDatasetCsv():
    result = collection.find_one(sort=[("timestamp", pymongo.DESCENDING)])
    path = result.get("pathDatasetCsv")
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