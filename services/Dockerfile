FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY config/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# App
COPY . /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8765

ENTRYPOINT ["python", "main.py"]
CMD ["--mode", "api"]
