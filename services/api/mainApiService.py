from fastapi import FastAPI, Request
import uvicorn
from localDatabase.collections.ApiAccessConfiguration.queries import getActualIpAndPort
from localDatabase.collections.DetectedAnomalies.queries import getLastAnomaliesSamples, getAllAnomaliesSamples
from localDatabase.collections.PredictionLogs.queries import getAllLogsSample, getLastLogsSample
from localDatabase.queues.queueSamples import QueueSamples
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import logging
import datetime
import os

app = FastAPI(middleware=[
    Middleware(CORSMiddleware, allow_origins=["*"])
])

queueSamples = QueueSamples()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/all_last_anomalies")
async def all_last_anomalies():
    result = getAllAnomaliesSamples()
    logging.info(result)
    if result:
        return {"anomalies": result}
    else:
        return "error", 500

@app.get("/last_anomalies/{fecha_inicio}/{fecha_fin}")
async def last_anomalies_by_dates(fecha_inicio: str, fecha_fin: str):
    # Convertir las cadenas de fecha a objetos de fecha de Python
    fecha_inicio_obj = None
    if fecha_inicio:
        fecha_inicio_obj = datetime.datetime.strptime(fecha_inicio, "%d/%m/%YT%H:%M")
    fecha_fin_obj = None
    if fecha_fin:
        fecha_fin_obj = datetime.datetime.strptime(fecha_fin, "%d/%m/%YT%H:%M")

    result = getLastAnomaliesSamples(fecha_inicio_obj, fecha_fin_obj)
    logging.info(result)
    if result:
        return {"anomalies": result}
    else:
        return "error", 500

@app.get("/all_logs_samples")
async def all_logs_samples():
    result = getAllLogsSample()
    logging.info(result)
    if result:
        return {"logs": result}
    else:
        return "error", 500

@app.get("/logs_samples/{fecha_inicio}/{fecha_fin}")
async def logs_samples_by_dates(fecha_inicio: str, fecha_fin: str):
    # Convertir las cadenas de fecha a objetos de fecha de Python
    fecha_inicio_obj = None
    if fecha_inicio:
        fecha_inicio_obj = datetime.datetime.strptime(fecha_inicio, "%d/%m/%YT%H:%M")
    fecha_fin_obj = None
    if fecha_fin:
        fecha_fin_obj = datetime.datetime.strptime(fecha_fin, "%d/%m/%YT%H:%M")

    result = getLastLogsSample(fecha_inicio_obj, fecha_fin_obj)
    logging.info(result)
    if result:
        return {"logs": result}
    else:
        return "error", 500


@app.post("/listen_sample")
async def listen_sample(info: Request):
    sampleJson = await info.json()
    logging.info(sampleJson)
    added = queueSamples.addSampleToQueue(sampleJson)
    logging.info(added)
    if added:
        return {"sample": "sample_received"}
    else:
        return {"error": added}, 500


if __name__ == "__main__":
    try:
        ip, port = getActualIpAndPort()
        uvicorn.run(app, host=ip, port=int(port), workers=4)
    except Exception as e:
        logging.exception("Error en API: " + str(e))
