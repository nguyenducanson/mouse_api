FROM python:3.7.11

ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY requirements.txt ./

RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6 -y && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . ./

ENV PYTHONPATH "${pwd}"
ENV PATH="${PATH}:/root/.local/bin"

CMD ["python3", "main.py"]
