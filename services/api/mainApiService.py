from fastapi import FastAPI, Request, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
import uvicorn
from localDatabase.collections.ApiAccessConfiguration.queries import getActualIpAndPort
from localDatabase.collections.DetectedAnomalies.queries import getLastAnomaliesSamples, getAllAnomaliesSamples
from localDatabase.collections.PredictionLogs.queries import getAllLogsSample, getLastLogsSample
from localDatabase.collections.TrainingEvaluationLogs.queries import getAllRetrainingEvaluations, getLastRetrainingEvaluations
from localDatabase.collections.BestTrainingEvaluationLogs import queries as BestTrainingQueries
from localDatabase.queues.queueSamples import QueueSamples
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import logging
import datetime
import json
import os
import pandas as pd
from ADS.ADSModelFactory import ADSModelFactory


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
        fecha_inicio_obj = datetime.datetime.strptime(fecha_inicio, "%d-%m-%YT%H:%M")
    fecha_fin_obj = None
    if fecha_fin:
        fecha_fin_obj = datetime.datetime.strptime(fecha_fin, "%d-%m-%YT%H:%M")

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
        fecha_inicio_obj = datetime.datetime.strptime(fecha_inicio, "%d-%m-%YT%H:%M")
    fecha_fin_obj = None
    if fecha_fin:
        fecha_fin_obj = datetime.datetime.strptime(fecha_fin, "%d-%m-%YT%H:%M")

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

@app.get("/all_logs_best_retraining_evaluation")
async def all_logs_best_retraining_evaluation():
    result = BestTrainingQueries.getAllRetrainingEvaluations()
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
        fecha_inicio_obj = datetime.datetime.strptime(fecha_inicio, "%d-%m-%YT%H:%M").timestamp()
    fecha_fin_obj = None
    if fecha_fin:
        fecha_fin_obj = datetime.datetime.strptime(fecha_fin, "%d-%m-%YT%H:%M").timestamp()

    result = getLastRetrainingEvaluations(fecha_inicio_obj, fecha_fin_obj)
    logging.info(result)
    if result:
        return {"logs": json.dumps(result, default=str)}
    else:
        return "error", 500

@app.get("/logs_best_retraining_evaluation/{fecha_inicio}/{fecha_fin}")
async def logs_best_retraining_evaluation(fecha_inicio: str, fecha_fin: str):
    # Convertir las cadenas de fecha a objetos de fecha de Python
    fecha_inicio_obj = None
    if fecha_inicio:
        fecha_inicio_obj = datetime.datetime.strptime(fecha_inicio, "%d-%m-%YT%H:%M").timestamp()
    fecha_fin_obj = None
    if fecha_fin:
        fecha_fin_obj = datetime.datetime.strptime(fecha_fin, "%d-%m-%YT%H:%M").timestamp()

    result = BestTrainingQueries.getLastRetrainingEvaluations(fecha_inicio_obj, fecha_fin_obj)
    logging.info(result)
    if result:
        return {"logs": json.dumps(result, default=str)}
    else:
        return "error", 500

@app.get("/actual_evaluation_dict")
async def actual_evaluation_dict():
    adsModel = ADSModelFactory.getADSModelVersion()
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
    adsModel = ADSModelFactory.getADSModelVersion()
    prediction = adsModel.predict_sample(sampleJson)
    sampleJson["prediction"] = prediction
    return {"sample": sampleJson}

@app.get("/get_samples_dataset_reviewed")
async def get_samples_dataset_reviewed():
    adsModel = ADSModelFactory.getADSModelVersion()
    datasetReviewedDataframe = adsModel.get_dataset_reviewed()
    samplesJson = []
    if datasetReviewedDataframe is not None:
        samplesJson = datasetReviewedDataframe.to_dict('records')
    return {"samples": samplesJson}

@app.post("/add_samples_dataset_reviewed")
async def add_samples_dataset_reviewed(info: Request):
    samplesJson = await info.json()
    adsModel = ADSModelFactory.getADSModelVersion()
    samplesResult = []
    for sampleJson in samplesJson:
        sampleJson = adsModel.add_sample_to_dataset(sampleJson, reviewed=True)
        samplesResult.append(sampleJson)
    return {"samples": samplesResult}

@app.post("/add_sample_dataset_reviewed")
async def add_sample_dataset_reviewed(info: Request):
    sampleJson = await info.json()
    adsModel = ADSModelFactory.getADSModelVersion()
    sampleJson = adsModel.add_sample_to_dataset(sampleJson, reviewed=True)
    return {"sample": sampleJson}

@app.post("/predict_multiple")
async def predict_multiple(info: Request):
    df_dict = await info.json()
    df = pd.read_json(df_dict)
    # df = pd.DataFrame(df_dict)
    adsModel = ADSModelFactory.getADSModelVersion()
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

    adsModel = ADSModelFactory.getADSModelVersion()
    evaluation_dict = adsModel.evaluate_dataframe(df)

    return {"evaluation_dict": evaluation_dict}

@app.get("/actual_model_json")
async def actual_model_json():
    adsModel = ADSModelFactory.getADSModelVersion()
    model_json = adsModel.get_actual_model_json()
    if model_json:
        return {"model_json": model_json}
    else:
        return "error", 500

@app.get("/actual_model_file")
async def actual_model_json():
    adsModel = ADSModelFactory.getADSModelVersion()
    model_path = adsModel.modelPath
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="El modelo no se encuentra disponible.")
    return FileResponse(model_path)

@app.post("/replace_actual_model")
async def replace_actual_model(model_bytes: UploadFile, info: Request):
    info_json = await info.form()
    model_bytes = await model_bytes.read()
    evaluation_dict = json.loads(info_json["evaluation_dict"])
    adsModel = ADSModelFactory.getADSModelVersion()
    # is_real_system = int(os.environ.get('IS_REAL_SYSTEM', 0))
    # if not is_real_system:
    #     raise HTTPException(status_code=403, detail="No es un sistema real")
    try:
        with open(adsModel.modelPath, "wb") as f:
            f.write(model_bytes)
        adsModel.__load_model__()
        adsModel.save_evaluation_model(evaluation_dict)
    except:
        raise HTTPException(status_code=500, detail="No se pudo actualizar el modelo.")
    return {"success": True}

if __name__ == "__main__":
    try:
        ip, port = getActualIpAndPort()
        uvicorn.run(app, host=ip, port=int(port), workers=4)
    except Exception as e:
        logging.exception("Error en API: " + str(e))
