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
