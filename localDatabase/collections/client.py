from pymongo import MongoClient
from localDatabase.collections.config import SERVER_URL, DATABASE

client = MongoClient(SERVER_URL)
db = client[DATABASE]
