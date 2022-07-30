#!/bin/bash

source commons.sh

JUPYTER_PORT=8888
SERVICE_NAME=jupyter-container

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
