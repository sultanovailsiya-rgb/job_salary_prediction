FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

WORKDIR /app

# Системные зависимости для компиляции пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Poetry и зависимости проекта
RUN pip install --no-cache-dir poetry
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --only main

# Копируем проект
COPY . .

# Создаем необходимые директории
RUN mkdir -p data results models

# Команда по умолчанию
CMD ["python", "-m", "src.main"]
