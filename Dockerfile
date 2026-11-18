
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY fastapi_app.py .

ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000

ENV MODEL_URI=models:/LogisticRegressionModelRegistered./3

EXPOSE 8000

CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
