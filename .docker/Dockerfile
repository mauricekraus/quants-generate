# syntax = docker/dockerfile:1.2
ARG KIND=3.10
FROM mkrausio/ml_research:torch-v2.0.1-3.10-cuda-11.8.0

ARG COMPOSE_PROJ_NAME="quants-generate"

ENV PROJECT_NAME=${COMPOSE_PROJ_NAME}
RUN echo ${COMPOSE_PROJ_NAME}/$PROJECT_NAME

RUN apt update -y && apt upgrade -y && apt install unzip ffmpeg libegl1 -y

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
  apt-get install -y git-lfs

RUN mkdir -p workspaces/$PROJECT_NAME
WORKDIR /workspaces/$PROJECT_NAME

COPY stmc/requirements.txt .

RUN python -m pip install --upgrade pip && \
  pip install "torch==2.0.1" -r requirements.txt

# CLIP's setup.py imports pkg_resources (needs a pre-81 setuptools) and
# chumpy's setup.py imports pip directly -- neither is present in pip's
# isolated build env, so build both against the outer, already-installed
# environment instead.
RUN pip install "setuptools<81" wheel && \
  pip install --no-build-isolation \
    "clip @ git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33" \
    "chumpy @ git+https://github.com/mauricekraus/chumpy.git"

ENV PYTHONPATH=$PYTHONPATH:src

CMD zsh
