from localDatabase.collections.client import db
import datetime
import pymongo

collection = db.PredictionLogs

def addPredictionLogSample(sampleJson):
    result = collection.insert_one(sampleJson)
    return result

def getAllLogsSample():
    result = collection.find().sort("timestamp", pymongo.DESCENDING)
    return result

def getLastLogsSample(fromDatetime, toDatetime=None):
    if not toDatetime:
        toDatetime = datetime.datetime.now().timestamp()
    query = {"timestamp": {"$gte": fromDatetime, "$lte": toDatetime}}
    result = collection.find(query).sort("timestamp", pymongo.DESCENDING)
    return result