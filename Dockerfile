FROM python:3.11-slim


WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install -e .

ENV REPO_TYPE=${REPO_TYPE:-json} \
    JSON_PATH=${JSON_PATH:-data.json} \
    PICKLE_PATH=${PICKLE_PATH:-data.pkl} \
    SQLITE_PATH=${SQLITE_PATH:-data.db} \
    PORT=${PORT:-8000}

EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
