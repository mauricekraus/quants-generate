# syntax = docker/dockerfile:1.2
ARG KIND=3.10
FROM mkrausio/ml_research:torch-v2.0.1-3.10-cuda-11.8.0

ARG COMPOSE_PROJ_NAME="tsqa_pipeline_v2"

ENV PROJECT_NAME=${COMPOSE_PROJ_NAME}
RUN echo ${COMPOSE_PROJ_NAME}/$PROJECT_NAME

RUN apt update -y && apt upgrade -y && apt install unzip ffmpeg libegl1 -y

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
  apt-get install -y git-lfs

RUN mkdir -p workspaces/$PROJECT_NAME
WORKDIR /workspaces/$PROJECT_NAME

COPY requirements.txt .

RUN \
  python -m pip install --upgrade pip && \
  pip install "torch==2.0.1" -r requirements.txt  


ENV PYTHONPATH=$PYTHONPATH:src

CMD zsh
