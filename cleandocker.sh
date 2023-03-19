docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)
docker rmi $(docker images -q)
sudo rm -rf /var/lib/docker/overlay2
sudo rm -rf /var/lib/docker
sudo rm -rf /var/lib/docker/tmp
sudo systemctl restart docker