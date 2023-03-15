import datetime
import pymongo

from localDatabase.collections.client import db

collection = db.TrainingEvaluationLogs

def addNewRetrainingEvaluation(data):
    data["timestamp"] = datetime.datetime.now().timestamp()
    result = collection.insert_one(data)
    return result

def getLastEvaluation():
    result = collection.find_one().sort("timestamp", pymongo.DESCENDING)
    return result