docker run --name mdv1-container -v /opt/tfm/OptimizationAutonomousVehiclesADS/datasets/nuImages/all/samples:/app/datasets -p 8001:8001 -p 27017:27017 -d mdv1
docker run --name mdv2-container -p 8001:8001 -p 27017:27017 -d mdv2

docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mdv2-container
