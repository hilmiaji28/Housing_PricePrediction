FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir \
    --timeout=1000 \
    --retries=5 \
    -r requirements.txt

COPY src/ ./src/
COPY Data/ ./Data/
COPY streamlit_app.py .

RUN mkdir -p mlruns logs models

EXPOSE 5000
EXPOSE 8501

ENV MODEL_VERSION=latest
ENV PYTHONUNBUFFERED=1