#!/bin/bash

# run tests
echo "***************************************************************************"
echo "[INFO] running python unittest"
python -m unittest discover "$CONTAINER_BASE_DIR"/ || exit 1

if [[ $MODE == 'prod' ]]; then
  echo "***************************************************************************"
  echo "[INFO] executing prod"
  python -u "$CONTAINER_BASE_DIR"/main.py
fi
