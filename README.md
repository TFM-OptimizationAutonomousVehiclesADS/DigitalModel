# DigitalModel

# Requisites
Docker:
sudo apt install docker.io

## Instalación

git clone https://github.com/TFM-OptimizationAutonomousVehiclesADS/DigitalModel.git

cd DigitalModel

## Construir Docker
sudo docker build -t jesuscumpli/model-digital:tag-version .

## Actualizar Docker
- sudo docker logs -f --tail 200 name-container // Ver logs del contenedor
- sudo docker exec -it name-container /bin/bash // Acceder al contenedor
- sudo docker commit <name-container jesuscumpli/model-digital:tag-version
- sudo docker stop name-container
- sudo docker rm name-container

## Push Docker
sudo docker image push jesuscumpli/model-digital:tag-version

## Limpiar Todo los datos de los Dockers
sudo ./cleandocker.sh
