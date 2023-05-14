import datetime
import pymongo
from localDatabase.collections.client import db

collection = db.BestTrainingEvaluationLogs

def addNewRetrainingEvaluation(data):
    data["timestamp"] = datetime.datetime.now().timestamp()
    result = collection.insert_one(data)
    return result

def getLastEvaluation():
    result = collection.find_one().sort("timestamp", pymongo.DESCENDING)
    return result

def getAllRetrainingEvaluations():
    result = list(collection.find().sort("timestamp", pymongo.DESCENDING))
    return result

def getLastRetrainingEvaluations(fromDatetime, toDatetime=None):
    if not toDatetime:
        toDatetime = datetime.datetime.now().timestamp()
    query = {"timestamp": {"$gte": fromDatetime, "$lte": toDatetime}}
    result = list(collection.find(query).sort("timestamp", pymongo.DESCENDING))
    return result