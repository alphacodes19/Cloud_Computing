# Keep the image small: CPU-only torch wheel (no CUDA libs), slim base, no model
# weights baked in (the service downloads them from object storage at startup).
FROM python:3.11-slim

WORKDIR /app

COPY service/requirements.txt .
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -r requirements.txt

COPY service/app.py .
COPY data/schema.json data/schema.json

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
