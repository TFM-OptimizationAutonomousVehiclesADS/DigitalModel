import redis

class RedisClient():
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if RedisClient.__instance == None:
            RedisClient()
        return RedisClient.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if RedisClient.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            RedisClient.__instance = self
            self.redis = redis.Redis(host='localhost', port=6379, db=0)


