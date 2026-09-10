FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r Requirements.txt

COPY MetaLib.py .

CMD ["python", "MetaLib.py"]