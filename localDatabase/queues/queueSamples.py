from localDatabase.queues.redisClient import RedisClient
import json

class QueueSamples():

    def __init__(self):
        self.queue = "SamplesQueue"
        self.redisClient = RedisClient.get_instance()

    def addSampleToQueue(self, sampleJson):
        sampleDumps = json.dumps(sampleJson)
        result = self.redisClient.redis.rpush(self.queue, sampleDumps)
        return result

    def getSampleFromQueue(self):
        sample = self.redisClient.redis.lpop(self.queue)
        if sample:
            sample = json.loads(sample)
        return sample

    def waitSampleFromQueue(self):
        queueReturned, sample = self.redisClient.redis.blpop(self.queue)
        if sample:
            sample = json.loads(sample)
        return sample