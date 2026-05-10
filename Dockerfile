FROM python:3.9-slim

WORKDIR /app

# Системные зависимости для компиляции пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем проект
COPY . .

# Создаем необходимые директории
RUN mkdir -p data results models

# Команда по умолчанию
CMD ["python", "src/main.py"]