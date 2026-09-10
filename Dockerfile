FROM python:3.12-slim

WORKDIR /app

COPY Requirements.txt .

RUN pip install --no-cache-dir -r Requirements.txt

COPY MetaLib.py .
COPY API.py .
COPY pyproject.toml .

CMD ["sh", "-c", "uvicorn API:app --host 0.0.0.0 --port ${PORT:-10000}"]
