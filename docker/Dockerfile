FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir TotalSegmentator

COPY app/ ./app/
COPY data/ ./data/

# Создание необходимых директорий
RUN mkdir -p data/input data/output data/models data/temp

# Установка переменных окружения
ENV PYTHONPATH=/app
ENV MODEL_TYPE=totalsegmentator

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]