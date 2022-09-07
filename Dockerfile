FROM python:3.8-slim as builder
RUN apt update && apt install --no-install-recommends -y build-essential gcc

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

COPY requirements.txt ./

RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6 -y && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

COPY . ./

ENV PYTHONUNBUFFERED 1

CMD ["python3", "main.py"]
