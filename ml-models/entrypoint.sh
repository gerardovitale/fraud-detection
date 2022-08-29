#!/bin/bash

# run tests
echo "***************************************************************************"
echo "[INFO] running python unittest"
python -m unittest discover "$CONTAINER_BASE_DIR"/ || exit 1

if [[ $MODE == 'exp' ]]; then
  echo "***************************************************************************"
  echo "[INFO] executing experiments"
  python -u "$CONTAINER_BASE_DIR"/main.py
fi
