FROM python:3.12-slim

WORKDIR /app

COPY Requirements.txt .

RUN pip install --no-cache-dir -r Requirements.txt

COPY MetaLib.py .
COPY pyproject.toml .

RUN pip install --no-cache-dir -e .

CMD ["metalib"]
