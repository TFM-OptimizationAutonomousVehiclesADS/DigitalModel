import logging
from celery import Celery
from celery.schedules import crontab, schedule
from ADS.ADSModel import ADSModel

app = Celery('retrainingceleryapp')
app.conf.beat_schedule = {
    'task-name': {
        'task': 'retrainingceleryapp.tasks.retraining_task',
        # 'schedule': crontab(hour='*/12'), # Ejecuta cada 12 horas
        'schedule': schedule(run_every='12'), # Ejecuta cada 12 segundos
    },
}

# Para ejecutar la pp
# celery -A serices.Retraining.retrainingceleryapp worker -B

@app.task
def retraining_task():
    logging.info("** RETRAINING TASK: Iniciando Modelo de Detección de Anomalías....")
    adsModel = ADSModel()
    logging.info("** RETRAINING TASK: Comenzando Reentrenamiento....")
    adsModel.retrain_model(random=True, size_split=2000)
    logging.info("** RETRAINING TASK: FIN REENTRENAMIENTO")