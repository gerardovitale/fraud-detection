#!/bin/bash

JUPYTER_PORT=8888
SERVICE_NAME=jupyter-container

dateTimeNow() {
  date +%m/%d/%YT%H:%M:%S
}

cleanUpDocker() {
  echo "$(dateTimeNow) - [INFO] - clean up docker containers and images"

  containerRunning=$(docker ps | grep "$1")
  if [[ $containerRunning ]]; then
    echo "$(dateTimeNow) - [INFO] - Stopping $1 container"
    docker container stop "$1"
  fi

  nodeContainer=$(docker container ls --all | grep "$1")
  if [[ $nodeContainer ]]; then
    echo "$(dateTimeNow) - [INFO] - Removing $1 container"
    docker container rm "$1"
  fi
}

runJupyterContainer() {
    cleanUpDocker $SERVICE_NAME

    echo "$(dateTimeNow) - [INFO] - Building new $SERVICE_NAME"
    docker build --platform linux/amd64 \
        -f notebooks/Dockerfile \
        --build-arg JUPYTER_PORT="$JUPYTER_PORT" \
        -t "$SERVICE_NAME" . || exit

    echo "$(dateTimeNow) - [INFO] - Running $SERVICE_NAME"
    docker run -d \
        --platform linux/amd64 \
        --name="$SERVICE_NAME" \
        -p "$JUPYTER_PORT":"$JUPYTER_PORT" \
        -v "$PWD"/notebooks/src:/app/src \
        -v "$PWD"/data:/app/data \
        --restart unless-stopped \
        "$SERVICE_NAME" || exit

    docker logs -f "$SERVICE_NAME"
}

runJupyterContainer
