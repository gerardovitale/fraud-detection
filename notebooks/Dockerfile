ARG PY_IMAGE_VARIANT=slim
ARG PYTHON_VERSION=3.9.5

FROM python:${PYTHON_VERSION}-${PY_IMAGE_VARIANT} AS python3

RUN apt-get update && \
    python -m venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH

COPY ./notebooks/requirements.txt ${CONTAINER_BASE_DIR}/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r ${CONTAINER_BASE_DIR}/requirements.txt && \
    pip install jupyter
    
FROM python:${PYTHON_VERSION}-${PY_IMAGE_VARIANT} AS build-image

ENV PYTHONUNBUFFERED 1

ARG CONTAINER_BASE_DIR=/app
ARG JUPYTER_PORT=8888
ARG DATA_URL
ARG TINI_VERSION=v0.6.0

ENV CONTAINER_BASE_DIR=${CONTAINER_BASE_DIR} \
    JUPYTER_PORT=${JUPYTER_PORT} \
    DATA_URL=${DATA_URL} \
    PATH=/opt/venv/bin:$PATH

COPY --from=python3 /opt/venv /opt/venv

ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini

WORKDIR ${CONTAINER_BASE_DIR}/
EXPOSE ${JUPYTER_PORT}

COPY ./notebooks/entrypoint.sh .

RUN chmod +x /usr/bin/tini && \
    chmod +x ./entrypoint.sh

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "./entrypoint.sh"]