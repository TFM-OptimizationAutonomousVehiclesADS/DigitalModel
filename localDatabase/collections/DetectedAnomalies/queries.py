from localDatabase.collections.client import db
import pymongo
import datetime

collection = db.DetectedAnomalies

def addAnomalySample(sampleJson):
    result = collection.insert_one(sampleJson)
    return result

def getAllAnomaliesSamples():
    result = list(collection.find().sort("timestamp", pymongo.DESCENDING))
    return result

def getLastAnomaliesSamples(fromDatetime, toDatetime=None):
    if not toDatetime:
        toDatetime = datetime.datetime.now().timestamp()
    query = {"timestamp": {"$gte": fromDatetime, "$lte": toDatetime}}
    result = list(collection.find(query).sort("timestamp", pymongo.DESCENDING))
    return result