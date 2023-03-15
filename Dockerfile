FROM python:3.8
COPY . /app
COPY requirements.txt /app

WORKDIR /app
ENV PYTHONPATH=/app

RUN wget -qO - https://www.mongodb.org/static/pgp/server-4.4.asc | apt-key add -
RUN echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/4.4 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-4.4.list
RUN apt-get update && apt-get install -y mongodb-org
RUN apt-get update && apt-get install -y graphviz libgraphviz-dev pkg-config

# Crear el directorio de datos
RUN mkdir -p /data/db && \
    chown -R mongodb:mongodb /data/db

RUN apt-get install -y redis-server

RUN pip install -r requirements.txt

# COPY /etc/redis/redis.conf /etc/redis/redis.conf

#COPY ./services/detection/ads.service /etc/systemd/system/
#COPY ./services/api/api.service /etc/systemd/system/
#COPY ./services/simulator/simulator.service /etc/systemd/system/

COPY start.sh /usr/local/bin/

CMD ["/bin/bash", "-c", "/usr/local/bin/start.sh"]
RUN chmod +x /usr/local/bin/start.sh

