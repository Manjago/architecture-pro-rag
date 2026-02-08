"""
update_index.py — Инкрементальное обновление FAISS-индекса.

Задание 6 проектной работы спринта 7.

Скрипт:
  1. Сканирует data/knowledge_base/ на новые или изменённые файлы
  2. Сравнивает с манифестом (список уже проиндексированных файлов)
  3. Для новых/изменённых файлов: чанкинг → эмбеддинги → добавление в FAISS
  4. Обновляет манифест и сохраняет индекс
  5. Логирует процесс в logs/update_log.jsonl

Использование:
    python scripts/update_index.py

    # Для автоматического запуска (cron):
    0 6 * * * cd /path/to/project && .venv/bin/python scripts/update_index.py

Зависимости:
    pip install langchain langchain-community langchain-text-splitters faiss-cpu sentence-transformers
"""

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Импортируем E5Embeddings из build_index
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_index import E5Embeddings, EMBEDDING_MODEL


# === Настройки ===

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..")

KB_DIR = os.path.join(ROOT_DIR, "data", "knowledge_base")
INDEX_DIR = os.path.join(ROOT_DIR, "faiss_index")
MANIFEST_PATH = os.path.join(INDEX_DIR, "manifest.json")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
LOG_PATH = os.path.join(LOG_DIR, "update_log.jsonl")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
EXTENSIONS = (".md", ".txt")


# === Утилиты ===

def file_hash(filepath: str) -> str:
    """Вычисляет SHA-256 хеш файла."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    """
    Загружает манифест — словарь уже проиндексированных файлов.
    Формат: { "filename.md": {"hash": "abc123", "mtime": 1234567890.0, "chunks": 15} }
    """
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict):
    """Сохраняет манифест."""
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def append_log(entry: dict):
    """Добавляет запись в лог-файл (JSONL)."""
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def find_changed_files(manifest: dict) -> tuple[list[str], list[str]]:
    """
    Сканирует KB_DIR и определяет новые/изменённые файлы.

    Returns:
        (new_files, modified_files) — списки имён файлов
    """
    new_files = []
    modified_files = []

    for fname in sorted(os.listdir(KB_DIR)):
        if not fname.endswith(EXTENSIONS):
            continue

        filepath = os.path.join(KB_DIR, fname)
        current_hash = file_hash(filepath)

        if fname not in manifest:
            new_files.append(fname)
        elif manifest[fname]["hash"] != current_hash:
            modified_files.append(fname)

    return new_files, modified_files


def find_deleted_files(manifest: dict) -> list[str]:
    """Находит файлы, которые были в манифесте, но удалены из KB_DIR."""
    deleted = []
    for fname in manifest:
        filepath = os.path.join(KB_DIR, fname)
        if not os.path.exists(filepath):
            deleted.append(fname)
    return deleted


# === Чанкинг и индексация ===

def chunk_file(filepath: str, filename: str) -> list:
    """
    Читает файл, разбивает на чанки, возвращает список (text, metadata).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )

    chunks = splitter.split_text(text)
    base_name = os.path.splitext(filename)[0]

    result = []
    for i, chunk_text in enumerate(chunks):
        metadata = {
            "source": filename,
            "chunk_id": f"{base_name}_{i:03d}",
        }
        result.append((chunk_text, metadata))

    return result


def update_index():
    """Основная функция инкрементального обновления индекса."""
    start_time = time.time()
    timestamp = datetime.now(timezone.utc).isoformat()

    print(f"[{timestamp}] Начало обновления индекса")
    print(f"  Источник: {os.path.abspath(KB_DIR)}")
    print(f"  Индекс:   {os.path.abspath(INDEX_DIR)}")

    # 1. Загружаем манифест
    manifest = load_manifest()
    print(f"  Файлов в манифесте: {len(manifest)}")

    # 2. Находим изменения
    new_files, modified_files = find_changed_files(manifest)
    deleted_files = find_deleted_files(manifest)

    files_to_process = new_files + modified_files

    print(f"\n  Новых файлов:       {len(new_files)}")
    print(f"  Изменённых файлов:  {len(modified_files)}")
    print(f"  Удалённых файлов:   {len(deleted_files)}")

    if not files_to_process and not deleted_files:
        print("\n  ✅ Индекс актуален, обновление не требуется.")
        log_entry = {
            "timestamp": timestamp,
            "status": "no_changes",
            "duration_sec": round(time.time() - start_time, 2),
            "new_files": 0,
            "modified_files": 0,
            "deleted_files": 0,
            "new_chunks": 0,
            "total_chunks": sum(m["chunks"] for m in manifest.values()),
            "errors": [],
        }
        append_log(log_entry)
        return

    # 3. Загружаем модель эмбеддингов
    print("\n  Загрузка модели эмбеддингов...")
    embeddings = E5Embeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 4. Загружаем существующий индекс (или создаём новый)
    if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        print("  Загрузка существующего индекса...")
        vectorstore = FAISS.load_local(
            INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )
    else:
        vectorstore = None

    # 5. Обрабатываем новые/изменённые файлы
    total_new_chunks = 0
    errors = []

    for fname in files_to_process:
        filepath = os.path.join(KB_DIR, fname)
        status = "новый" if fname in new_files else "изменён"

        try:
            chunks = chunk_file(filepath, fname)
            if not chunks:
                print(f"    ⚠️  {fname} — пустой файл, пропущен")
                continue

            texts = [c[0] for c in chunks]
            metadatas = [c[1] for c in chunks]

            if vectorstore is None:
                vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
            else:
                vectorstore.add_texts(texts, metadatas=metadatas)

            # Обновляем манифест
            manifest[fname] = {
                "hash": file_hash(filepath),
                "mtime": os.path.getmtime(filepath),
                "chunks": len(chunks),
            }

            total_new_chunks += len(chunks)
            print(f"    ✅ {fname} ({status}): {len(chunks)} чанков")

        except Exception as e:
            error_msg = f"{fname}: {str(e)}"
            errors.append(error_msg)
            print(f"    ❌ {error_msg}")

    # 6. Обрабатываем удалённые файлы
    # Примечание: FAISS не поддерживает удаление отдельных векторов.
    # Удаляем из манифеста; для полной очистки нужна пересборка (build_index.py).
    for fname in deleted_files:
        del manifest[fname]
        print(f"    🗑️  {fname} — удалён из манифеста (чанки останутся в индексе до пересборки)")

    # 7. Сохраняем индекс и манифест
    if vectorstore is not None:
        vectorstore.save_local(INDEX_DIR)
        print(f"\n  💾 Индекс сохранён: {os.path.abspath(INDEX_DIR)}")

    save_manifest(manifest)

    # 8. Итоги
    duration = round(time.time() - start_time, 2)
    total_chunks = sum(m["chunks"] for m in manifest.values())

    print(f"\n  📊 Итого:")
    print(f"     Новых чанков:      {total_new_chunks}")
    print(f"     Всего в манифесте: {total_chunks} чанков из {len(manifest)} файлов")
    print(f"     Ошибок:            {len(errors)}")
    print(f"     Время:             {duration} сек")

    # 9. Логируем
    log_entry = {
        "timestamp": timestamp,
        "status": "updated",
        "duration_sec": duration,
        "new_files": len(new_files),
        "modified_files": len(modified_files),
        "deleted_files": len(deleted_files),
        "new_chunks": total_new_chunks,
        "total_chunks": total_chunks,
        "errors": errors,
    }
    append_log(log_entry)

    if errors:
        print(f"\n  ⚠️  Были ошибки: {errors}")
    else:
        print(f"\n  ✅ Обновление завершено успешно!")


# === Main ===

if __name__ == "__main__":
    update_index()
