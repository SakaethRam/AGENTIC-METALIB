FROM python:3.12-slim

WORKDIR /app

COPY Requirements.txt .

RUN pip install --no-cache-dir -r Requirements.txt

COPY MetaLib.py .
COPY api.py .
COPY pyproject.toml .

RUN pip install --no-cache-dir -e .

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-10000}"]
