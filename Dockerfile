FROM nvidia/cuda:12.8.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      wget \
      gnupg2 \
      build-essential \
      curl \
      git \
      libsndfile1 \
      ffmpeg \
      unzip \
      python3 python3-venv python3-distutils python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt
RUN python3 -m pip install git+https://github.com/xhinker/sd_embed.git@main

COPY ./src /app/src

EXPOSE 5700

CMD ["python3", "-m", "src.server"]
