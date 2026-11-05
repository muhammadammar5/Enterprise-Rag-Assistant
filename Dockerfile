FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only PyTorch keeps the image far smaller than the default CUDA build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY eval ./eval

EXPOSE 8000

# Persist the vector DB and cache by mounting volumes (see README)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
