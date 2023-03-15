from localDatabase.queues.queueSamples import addSampleToQueue, getSampleFromQueue, waitSampleFromQueue


# addSampleToQueue({"my_dict": "hola"})
# addSampleToQueue("hola1")
# addSampleToQueue("hola2")

sample = waitSampleFromQueue()
print(sample)