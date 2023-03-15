import pymongo
import datetime
from localDatabase.collections.client import db

collection = db.ApiAccessConfiguration

def setActualIpAndPort(ip, port):
    result = {}
    result["ip"] = ip
    result["port"] = port
    result["timestamp"] = datetime.datetime.now().timestamp()
    result = collection.insert_one(result)
    return result

def getActualIpAndPort():
    result = collection.find_one(sort=[("timestamp", pymongo.DESCENDING)])
    ip = result.get("ip")
    port = result.get("port")
    return ip, port