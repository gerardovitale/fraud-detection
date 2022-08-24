#!/bin/bash

source commons.sh

SERVICE_NAME=api-container
MODE=dev
if [[ $1 != '' ]]; then
    MODE=$1
fi

runAPI() {
    cleanUpDocker $SERVICE_NAME

    echo "$(dateTimeNow) - [INFO] - Building new $SERVICE_NAME"
    docker build --platform linux/amd64 \
        -f api/Dockerfile \
        --build-arg MODE="$MODE" \
        -t "$SERVICE_NAME" . || exit

    echo "$(dateTimeNow) - [INFO] - Running $SERVICE_NAME"
    docker run -d \
        --platform linux/amd64 \
        --name="$SERVICE_NAME" \
        -p 8080:8080 \
        -v "$PWD"/data/ml_info.log:/app/data/ml_info.log \
        "$SERVICE_NAME" || exit

    docker logs -f "$SERVICE_NAME"
}

runAPI