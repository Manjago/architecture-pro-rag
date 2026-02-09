"""
build_index.py — Создание векторного индекса FAISS из базы знаний «МегаОфис».

Задание 3 проектной работы спринта 7.

Пайплайн:
  1. Загрузка .md файлов из data/knowledge_base/
  2. Разбиение на чанки (RecursiveCharacterTextSplitter)
  3. Создание эмбеддингов (intfloat/multilingual-e5-large)
  4. Сохранение FAISS-индекса в faiss_index/
  5. Верификация: тестовые запросы к индексу

Использование:
    python scripts/build_index.py

Зависимости:
    pip install langchain langchain-community sentence-transformers faiss-cpu
"""

import os
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# === Настройки ===

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..")
KB_DIR = os.path.join(ROOT_DIR, "data", "knowledge_base")
INDEX_DIR = os.path.join(ROOT_DIR, "faiss_index")

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
EMBEDDING_DIMENSION = 1024

CHUNK_SIZE = 800        # символов (~400-550 токенов для кириллицы)
CHUNK_OVERLAP = 150     # ~19% перекрытия

# Тестовые запросы для верификации индекса
TEST_QUERIES = [
    "Кто такой Токсичный Менеджер?",
    "Что такое Офис Гибели?",
    "Как работает Продуктивность?",
]


# === Кастомная обёртка для E5 эмбеддингов ===

class E5Embeddings(HuggingFaceEmbeddings):
    """
    Обёртка над HuggingFaceEmbeddings для моделей семейства E5.

    Модели E5 требуют префиксов:
      - "passage: " для документов (при индексации)
      - "query: "   для запросов (при поиске)

    Без этих префиксов качество поиска значительно хуже.
    Подробнее: https://huggingface.co/intfloat/multilingual-e5-large
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Добавляет префикс 'passage: ' к каждому документу."""
        prefixed = [f"passage: {text}" for text in texts]
        return super().embed_documents(prefixed)

    def embed_query(self, text: str) -> list[float]:
        """Добавляет префикс 'query: ' к запросу."""
        return super().embed_query(f"query: {text}")


# === Функции пайплайна ===

def load_documents(kb_dir: str) -> list[Document]:
    """
    Загружает .md файлы из директории и создаёт Document с метаданными.

    Метаданные каждого документа:
      - source: имя файла
      - title: заголовок (из первой строки '# ...')
    """
    documents = []
    files = sorted(f for f in os.listdir(kb_dir) if f.endswith(".md"))

    if not files:
        print(f"✗ Нет .md файлов в {os.path.abspath(kb_dir)}")
        print("  Сначала запустите скрипты подготовки данных.")
        return []

    for filename in files:
        filepath = os.path.join(kb_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Извлекаем заголовок из первой строки (формат: "# Название статьи")
        title = filename.replace(".md", "").replace("_", " ")
        first_line = content.split("\n", 1)[0].strip()
        if first_line.startswith("# "):
            title = first_line[2:].strip()

        doc = Document(
            page_content=content,
            metadata={"source": filename, "title": title},
        )
        documents.append(doc)

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Разбивает документы на чанки с сохранением метаданных.

    Каждый чанк получает дополнительное поле metadata['chunk_id']
    в формате 'filename_NNN'.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Добавляем chunk_id к метаданным
    # Группируем по source, чтобы нумерация была в пределах файла
    source_counters: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        idx = source_counters.get(source, 0)
        chunk.metadata["chunk_id"] = f"{source.replace('.md', '')}_{idx:03d}"
        source_counters[source] = idx + 1

    return chunks


def create_embeddings() -> E5Embeddings:
    """
    Инициализирует модель эмбеддингов E5.

    Первый запуск скачает модель (~2.2 GB) из HuggingFace.
    """
    print(f"Загрузка модели эмбеддингов: {EMBEDDING_MODEL}")
    print("  (первый запуск может занять несколько минут — скачивается модель ~2.2 GB)")

    embeddings = E5Embeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return embeddings


def build_faiss_index(
    chunks: list[Document], embeddings: E5Embeddings
) -> FAISS:
    """Создаёт FAISS-индекс из чанков."""
    print(f"\nСоздание FAISS-индекса из {len(chunks)} чанков...")
    print("  (это может занять несколько минут на CPU)")

    start = time.time()
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    elapsed = time.time() - start

    print(f"  Индекс создан за {elapsed:.1f} сек.")
    return vectorstore


def verify_index(vectorstore: FAISS, queries: list[str]):
    """
    Верификация индекса: тестовые запросы с выводом результатов.
    """
    print(f"\n{'='*60}")
    print("ВЕРИФИКАЦИЯ ИНДЕКСА")
    print(f"{'='*60}")

    for i, query in enumerate(queries, 1):
        print(f"\n--- Запрос {i}: «{query}» ---\n")

        results = vectorstore.similarity_search_with_score(query, k=3)

        for rank, (doc, score) in enumerate(results, 1):
            # Обрезаем текст для читаемости
            preview = doc.page_content[:200].replace("\n", " ")
            print(f"  [{rank}] score={score:.4f}")
            print(f"      Источник: {doc.metadata.get('source', '?')}")
            print(f"      Чанк:    {doc.metadata.get('chunk_id', '?')}")
            print(f"      Текст:   {preview}...")
            print()


def print_stats(documents: list[Document], chunks: list[Document]):
    """Выводит статистику по индексу."""
    print(f"\n{'='*60}")
    print("СТАТИСТИКА")
    print(f"{'='*60}")
    print(f"  Модель эмбеддингов:  {EMBEDDING_MODEL}")
    print(f"  Размерность:         {EMBEDDING_DIMENSION}")
    print(f"  Параметры чанкинга:  chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    print(f"  Файлов загружено:    {len(documents)}")
    print(f"  Чанков в индексе:    {len(chunks)}")

    # Распределение чанков по файлам
    chunks_per_file: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata["source"]
        chunks_per_file[src] = chunks_per_file.get(src, 0) + 1

    avg = len(chunks) / len(documents) if documents else 0
    min_chunks = min(chunks_per_file.values()) if chunks_per_file else 0
    max_chunks = max(chunks_per_file.values()) if chunks_per_file else 0
    print(f"  Чанков на файл:      min={min_chunks}, max={max_chunks}, avg={avg:.1f}")

    # Оценка памяти
    mem_mb = len(chunks) * EMBEDDING_DIMENSION * 4 / (1024 * 1024)
    print(f"  Оценка RAM индекса:  ~{mem_mb:.1f} MB")


# === Main ===

def main():
    # 1. Загрузка документов
    print(f"Загрузка документов из: {os.path.abspath(KB_DIR)}\n")
    documents = load_documents(KB_DIR)
    if not documents:
        return
    print(f"  Загружено файлов: {len(documents)}\n")

    # 2. Чанкинг
    print("Разбиение на чанки...")
    chunks = split_documents(documents)
    print(f"  Получено чанков: {len(chunks)}\n")

    # 3. Эмбеддинги + FAISS-индекс
    embeddings = create_embeddings()
    vectorstore = build_faiss_index(chunks, embeddings)

    # 4. Сохранение
    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)
    print(f"\nИндекс сохранён в: {os.path.abspath(INDEX_DIR)}")

    # 5. Статистика
    print_stats(documents, chunks)

    # 6. Верификация
    verify_index(vectorstore, TEST_QUERIES)

    print(f"\n{'='*60}")
    print("Готово! Индекс создан и верифицирован.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
