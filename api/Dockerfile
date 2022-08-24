ARG PY_IMAGE_VARIANT=slim
ARG PYTHON_VERSION=3.9.5

FROM python:${PYTHON_VERSION}-${PY_IMAGE_VARIANT} AS python3

RUN apt-get update && \
    python -m venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH

COPY ./api/requirements.txt ${CONTAINER_BASE_DIR}/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r ${CONTAINER_BASE_DIR}/requirements.txt

FROM python:${PYTHON_VERSION}-${PY_IMAGE_VARIANT} AS build-image

ENV PYTHONUNBUFFERED 1

ARG CONTAINER_BASE_DIR=/app
ARG MODE

ENV CONTAINER_BASE_DIR=${CONTAINER_BASE_DIR} \
    PATH=/opt/venv/bin:$PATH \
    MODE=${MODE}

EXPOSE 8080

COPY --from=python3 /opt/venv /opt/venv

WORKDIR ${CONTAINER_BASE_DIR}/

COPY ./api/src ${CONTAINER_BASE_DIR}/
COPY ./api/entrypoint.sh .

RUN chmod +x ./entrypoint.sh

ENTRYPOINT ["bash", "./entrypoint.sh", "${MODE}"]
