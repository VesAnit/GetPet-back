# Используем официальный образ Python версии 3.9
FROM python:3.12

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN mkdir -p uploads


EXPOSE 8080


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
