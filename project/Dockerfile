FROM python:3.11-slim

WORKDIR /app

# Копируем зависимости и устанавливаем
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Открываем порт Flask
EXPOSE 5000

# Запуск из папки /app (=project/)
CMD ["python", "-X", "utf8", "src/run_webui.py"]
