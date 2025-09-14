# Depend on your device, please check: https://hub.docker.com/r/nvidia/cuda/tags
# Find your cuda versio with cuda:xx.x.x-(devel-)ubuntu22.04
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/* 

# Set workdir
WORKDIR /app

# Copy app code and install deps
COPY app/ /app/
COPY ./requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "12999"]
