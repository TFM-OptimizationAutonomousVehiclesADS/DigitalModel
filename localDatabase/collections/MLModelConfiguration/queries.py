from localDatabase.collections.client import db
import pymongo
import datetime

collection = db.MLModelConfiguration

def setModelConfig(config):
    config["timestamp"] = datetime.datetime.now().timestamp()
    result = collection.insert_one(config)
    return result

def getImageModelPath():
    result = collection.find_one(sort=[("timestamp", pymongo.DESCENDING)])
    path = result.get("imageModelPath")
    return path

def getEvaluationModelPath():
    result = collection.find_one(sort=[("timestamp", pymongo.DESCENDING)])
    path = result.get("evaluationModelPath")
    return path

def getModelPath():
    result = collection.find_one(sort=[("timestamp", pymongo.DESCENDING)])
    path = result.get("modelPath")
    return path

def getFeatures():
    result = collection.find_one(sort=[("timestamp", pymongo.DESCENDING)])
    features = result.get("features")
    return features

def getImagesFeatures():
    result = collection.find_one(sort=[("timestamp", pymongo.DESCENDING)])
    features = result.get("imagesFeatures")
    return features

def getFloatFeatures():
    result = collection.find_one(sort=[("timestamp", pymongo.DESCENDING)])
    features = result.get("floatFeatures")
    return features

def getSizeImage():
    result = collection.find_one(sort=[("timestamp", pymongo.DESCENDING)])
    sizeImageWidth = int(result.get("sizeImageWidth").get("$numberLong"))
    sizeImageHeight = int(result.get("sizeImageHeight").get("$numberLong"))
    return [sizeImageHeight, sizeImageWidth]

def getThreshold():
    result = collection.find_one(sort=[("timestamp", pymongo.DESCENDING)])
    threshold = result.get("threshold")
    return threshold

def getModelCompilerOptimizer():
    result = collection.find_one(sort=[("timestamp", pymongo.DESCENDING)])
    compileOptimizer = result.get("compileOptimizer")
    if compileOptimizer:
        return compileOptimizer.get("optimizer")
    return None

def getModelCompilerLoss():
    result = collection.find_one(sort=[("timestamp", pymongo.DESCENDING)])
    compileOptimizer = result.get("compileOptimizer")
    if compileOptimizer:
        return compileOptimizer.get("loss")
    return None