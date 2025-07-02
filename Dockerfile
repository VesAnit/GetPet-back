# Используем официальный образ Python версии 3.9
FROM python:3.12

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем зависимости для psycopg2-binary
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Копируем весь код проекта
COPY . .

# Создаём папку uploads, которая нужна для твоего приложения
RUN mkdir -p uploads

# Указываем порт, который будет использовать приложение
EXPOSE 8080

# Команда для запуска FastAPI с uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
