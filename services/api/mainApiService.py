from fastapi import FastAPI, Request
import uvicorn
from localDatabase.collections.ApiAccessConfiguration.queries import getActualIpAndPort
from localDatabase.collections.DetectedAnomalies.queries import getLastAnomaliesSamples, getAllAnomaliesSamples
from localDatabase.collections.PredictionLogs.queries import getAllLogsSample, getLastLogsSample
from localDatabase.collections.TrainingEvaluationLogs.queries import getAllRetrainingEvaluations, getLastRetrainingEvaluations
from localDatabase.queues.queueSamples import QueueSamples
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import logging
import datetime
from ADS.ADSModel import ADSModel
import json
import os
import pandas as pd

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
        return {"anomalies": json.dumps(result, default=str)}
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
        return {"anomalies": json.dumps(result, default=str)}
    else:
        return "error", 500

@app.get("/all_logs_samples")
async def all_logs_samples():
    result = getAllLogsSample()
    logging.info(result)
    if result:
        return {"logs": json.dumps(result, default=str)}
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
        return {"logs": json.dumps(result, default=str)}
    else:
        return "error", 500

@app.get("/all_logs_retraining_evaluation")
async def all_logs_retraining_evaluation():
    result = getAllRetrainingEvaluations()
    logging.info(result)
    if result:
        return {"logs": json.dumps(result, default=str)}
    else:
        return "error", 500

@app.get("/logs_retraining_evaluation/{fecha_inicio}/{fecha_fin}")
async def logs_retraining_evaluation(fecha_inicio: str, fecha_fin: str):
    # Convertir las cadenas de fecha a objetos de fecha de Python
    fecha_inicio_obj = None
    if fecha_inicio:
        fecha_inicio_obj = datetime.datetime.strptime(fecha_inicio, "%d/%m/%YT%H:%M")
    fecha_fin_obj = None
    if fecha_fin:
        fecha_fin_obj = datetime.datetime.strptime(fecha_fin, "%d/%m/%YT%H:%M")

    result = getLastRetrainingEvaluations(fecha_inicio_obj, fecha_fin_obj)
    logging.info(result)
    if result:
        return {"logs": json.dumps(result, default=str)}
    else:
        return "error", 500

@app.get("/actual_evaluation_dict")
async def actual_evaluation_dict():
    adsModel = ADSModel()
    evaluation_dict = adsModel.get_actual_evaluation_model()
    if evaluation_dict:
        return {"evaluation_dict": evaluation_dict}
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

@app.post("/predict_single")
async def predict_single(info: Request):
    sampleJson = await info.json()
    adsModel = ADSModel()
    prediction = adsModel.predict_sample(sampleJson)
    sampleJson["prediction"] = prediction
    return {"sample": sampleJson}

@app.post("/add_sample_dataset_reviewed")
async def add_sample_dataset_reviewed(info: Request):
    sampleJson = await info.json()
    adsModel = ADSModel()
    sampleJson = adsModel.add_sample_to_dataset(sampleJson, reviewed=True)
    return {"sample": sampleJson}

@app.post("/predict_multiple")
async def predict_multiple(info: Request):
    df_dict = await info.json()
    df = pd.read_json(df_dict)
    # df = pd.DataFrame(df_dict)
    adsModel = ADSModel()
    samplesJson = []
    for index, row in df.iterrows():
        sampleJson = row.to_dict()
        prediction = adsModel.predict_sample(sampleJson)
        sampleJson["prediction"] = prediction
        samplesJson.append(sampleJson)
    return {"samples": samplesJson}

@app.post("/evaluate_dataframe")
async def evaluate_dataframe(info: Request):
    df_dict = await info.json()
    df = pd.read_json(df_dict)

    adsModel = ADSModel()
    evaluation_dict = adsModel.evaluate_dataframe(df)

    return {"evaluation_dict": evaluation_dict}

if __name__ == "__main__":
    try:
        ip, port = getActualIpAndPort()
        uvicorn.run(app, host=ip, port=int(port), workers=4)
    except Exception as e:
        logging.exception("Error en API: " + str(e))
