from localDatabase.queues.queueSamples import QueueSamples
import pandas as pd
import requests

SLEEP_TIME = 10
queueSamples = QueueSamples()


queueSamples.addSampleToQueue({"my_dict": "hola2"})
# addSampleToQueue("hola1")
# addSampleToQueue("hola2")

# sample = getSampleFromQueue()
# print(sample)