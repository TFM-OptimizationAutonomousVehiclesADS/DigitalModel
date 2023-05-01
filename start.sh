#!/bin/bash

# Iniciar MongoDB y Redis
mongod --logpath /var/log/mongodb.log &
service redis-server start &
sleep 5

python3 /app/localDatabase/collections/init_mongo.py > /proc/1/fd/1 2>&1
sleep 5

python3 /app/ADS/initTraining.py > /proc/1/fd/1 2>&1
sleep 5

# Iniciar los servicios de Python
#uvicorn services.api.mainApiService:app --host 0.0.0.0 --port 8001 &
#sleep 5
#
#python3 /app/services/detection/mainDetectionService.py > /proc/1/fd/1 2>&1 &
#python3 /app/services/simulator/mainRandomSamplesGeneratorService.py > /proc/1/fd/1 2>&1 &
#if [ -n "${IS_REAL_SYSTEM+x}" ] && [ "$IS_REAL_SYSTEM" = "1" ]; then
#    echo "IS_REAL_SYSTEM is set to 1. Skipping mainRetrainingService.py."
#else
#    python3 /app/services/retraining/mainRetrainingService.py > /proc/1/fd/1 2>&1 &
#fi

systemctl start api.service
sleep 5
systemctl start ads.service
systemctl start simulator.service
sleep 5
if [ -n "${IS_REAL_SYSTEM+x}" ] && [ "$IS_REAL_SYSTEM" = "1" ]; then
    echo "IS_REAL_SYSTEM is set to 1. Skipping mainRetrainingService.py."
else
    systemctl start retraining.service
fi

tail -f /dev/null