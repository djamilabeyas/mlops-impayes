FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 MODEL_PATH=/app/models/model.joblib
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY src ./src
COPY params.yaml .
COPY models ./models
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
