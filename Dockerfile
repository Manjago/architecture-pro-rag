FROM python:3.11-slim

WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python-зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код и данные
COPY scripts/ ./scripts/
COPY data/knowledge_base/ ./data/knowledge_base/
COPY faiss_index/ ./faiss_index/
COPY golden_questions.json .
COPY terms_map.json .

# Порт для будущего REST API (опционально)
EXPOSE 8000

# По умолчанию — консольный бот
CMD ["python", "scripts/rag_bot.py"]
