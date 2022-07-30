#!/bin/bash

## run tests
#echo "***************************************************************************"
#echo "[INFO] running python unittest"
#python -m unittest discover "$CONTAINER_BASE_DIR"/ || exit 1
#
#if [[ $MODE == 'pipe' ]]; then
#  # execute models
#  echo "***************************************************************************"
#  echo "[INFO] executing models"
#  python -u "$CONTAINER_BASE_DIR"/main.py
#fi

python -u "$CONTAINER_BASE_DIR"/main.py