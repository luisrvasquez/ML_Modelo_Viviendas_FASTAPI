FROM python:3.10-slim

RUN apt-get update && apt-get install -y build-essential gcc wget git && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

COPY app/ /app

COPY requirements.txt .

RUN pip install -r requirements.txt

# Comando para ejecutar la API
CMD ["gunicorn", "app.main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]



