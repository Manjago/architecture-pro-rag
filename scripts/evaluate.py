"""
evaluate.py — Автоматическое тестирование RAG-бота на золотом наборе вопросов.

Задание 7 проектной работы спринта 7.

Скрипт:
  1. Загружает FAISS-индекс и LLM
  2. Прогоняет вопросы из golden_questions.json
  3. Оценивает каждый ответ (успех / отказ / ложный результат)
  4. Логирует в logs/evaluation_log.jsonl
  5. Выводит сводный отчёт

Использование:
    python scripts/evaluate.py

Предварительно:
    1. Удалить 2-3 файла из knowledge_base/ (создать пробелы)
    2. Пересобрать индекс: python scripts/build_index.py
    3. Убедиться, что Ollama запущена: ollama serve

Зависимости:
    pip install langchain langchain-community langchain-ollama faiss-cpu sentence-transformers
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# Импортируем компоненты из существующих скриптов
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_index import E5Embeddings, EMBEDDING_MODEL
from rag_bot import (
    SYSTEM_PROMPT_PROTECTED,
    FEW_SHOT_EXAMPLES,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_NUM_CTX,
    TOP_K,
    format_context,
    build_user_message,
    is_chunk_safe,
)


# === Настройки ===

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..")
INDEX_DIR = os.path.join(ROOT_DIR, "faiss_index")
GOLDEN_PATH = os.path.join(ROOT_DIR, "golden_questions.json")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
EVAL_LOG_PATH = os.path.join(LOG_DIR, "evaluation_log.jsonl")

# Паттерны отказа
REFUSAL_PATTERNS = [
    "нет информации",
    "не знаю",
    "не найден",
    "не могу ответить",
    "недостаточно",
    "отсутствует",
    "нет данных",
    "не содержит",
]


# === Оценка ответа ===

def is_refusal(answer: str) -> bool:
    """Проверяет, является ли ответ отказом."""
    answer_lower = answer.lower()
    for pattern in REFUSAL_PATTERNS:
        if pattern in answer_lower:
            return True
    return False


def check_keywords(answer: str, keywords: list[str]) -> list[str]:
    """Возвращает список найденных ключевых слов в ответе."""
    answer_lower = answer.lower()
    found = []
    for kw in keywords:
        if kw.lower() in answer_lower:
            found.append(kw)
    return found


def evaluate_answer(answer: str, question: dict) -> dict:
    """
    Оценивает ответ бота.

    Returns:
        {
            "status": "TP" | "TN" | "FP" | "FN",
            "explanation": "...",
            "is_correct": True | False
        }

    Матрица:
        TP (True Positive) — ожидали ответ, бот ответил с ключевыми словами
        TN (True Negative) — ожидали отказ, бот отказался
        FP (False Positive) — ожидали отказ, бот ответил (утечка или галлюцинация)
        FN (False Negative) — ожидали ответ, бот отказался (пробел в базе)
    """
    expected_type = question["type"]
    refusal = is_refusal(answer)
    found_kw = check_keywords(answer, question.get("expected_keywords", []))

    if expected_type == "success":
        if not refusal and found_kw:
            return {
                "status": "TP",
                "explanation": f"Ответил корректно, найдены ключевые слова: {found_kw}",
                "is_correct": True,
            }
        elif refusal:
            return {
                "status": "FN",
                "explanation": "Ожидался ответ, но бот отказался (пробел в базе?)",
                "is_correct": False,
            }
        else:
            return {
                "status": "FN",
                "explanation": f"Ответил, но без ключевых слов. Ожидались: {question['expected_keywords']}",
                "is_correct": False,
            }
    else:  # expected_type == "refusal"
        if refusal:
            return {
                "status": "TN",
                "explanation": "Корректный отказ — информации нет в базе",
                "is_correct": True,
            }
        else:
            return {
                "status": "FP",
                "explanation": "Ожидался отказ, но бот ответил (галлюцинация или утечка?)",
                "is_correct": False,
            }


# === RAG-пайплайн с логированием ===

def ask_with_logging(
    question: str,
    vectorstore: FAISS,
    llm: ChatOllama,
) -> dict:
    """
    RAG-пайплайн с полным логированием.

    Returns:
        {
            "question": str,
            "answer": str,
            "timestamp": str,
            "chunks_found": bool,
            "chunks_count": int,
            "answer_length": int,
            "sources": [str],
            "scores": [float],
            "filtered_count": int,
            "duration_sec": float,
        }
    """
    start = time.time()
    timestamp = datetime.now(timezone.utc).isoformat()

    # 1. Поиск чанков
    results = vectorstore.similarity_search_with_score(question, k=TOP_K)

    # 2. Фильтрация (режим full)
    doc_objects = []
    filtered_count = 0
    all_sources = []
    all_scores = []

    for doc, score in results:
        source = doc.metadata.get("source", "?")
        all_sources.append(source)
        all_scores.append(round(float(score), 4))

        if not is_chunk_safe(doc.page_content):
            filtered_count += 1
            continue
        doc_objects.append(doc)

    # 3. Если все чанки отфильтрованы
    if not doc_objects:
        answer = "К сожалению, все найденные документы были отфильтрованы."
    else:
        # 4. Сборка промпта и генерация
        context = format_context(doc_objects)
        user_message = build_user_message(context, question)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT_PROTECTED),
            HumanMessage(content=user_message),
        ]

        response = llm.invoke(messages)
        answer = response.content

    duration = round(time.time() - start, 2)

    return {
        "question": question,
        "answer": answer,
        "timestamp": timestamp,
        "chunks_found": len(doc_objects) > 0,
        "chunks_count": len(doc_objects),
        "answer_length": len(answer),
        "sources": list(dict.fromkeys(all_sources)),  # уникальные, с сохранением порядка
        "scores": all_scores,
        "filtered_count": filtered_count,
        "duration_sec": duration,
    }


# === Основной процесс ===

def run_evaluation():
    """Запускает полный цикл оценки."""
    print("=" * 60)
    print("Автоматическая оценка RAG-бота")
    print("=" * 60)

    # 1. Загружаем golden questions
    if not os.path.exists(GOLDEN_PATH):
        print(f"❌ Файл {GOLDEN_PATH} не найден!")
        sys.exit(1)

    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"\n  Загружено вопросов: {len(questions)}")
    print(f"    Ожидается ответов: {sum(1 for q in questions if q['type'] == 'success')}")
    print(f"    Ожидается отказов: {sum(1 for q in questions if q['type'] == 'refusal')}")

    # 2. Загружаем индекс и LLM
    print("\n  Загрузка модели эмбеддингов...")
    embeddings = E5Embeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"  Загрузка FAISS-индекса из {INDEX_DIR}...")
    vectorstore = FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )

    print(f"  Подключение к Ollama ({OLLAMA_MODEL})...")
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=OLLAMA_TEMPERATURE,
        num_ctx=OLLAMA_NUM_CTX,
    )

    # 3. Прогон вопросов
    print(f"\n{'─' * 60}")
    print("Начинаю тестирование...\n")

    results = []
    os.makedirs(LOG_DIR, exist_ok=True)

    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q['question']}")

        # Вызов RAG с логированием
        log_entry = ask_with_logging(q["question"], vectorstore, llm)

        # Оценка
        evaluation = evaluate_answer(log_entry["answer"], q)

        # Формируем полную запись
        full_entry = {
            **log_entry,
            "question_id": q["id"],
            "expected_type": q["type"],
            "topic": q["topic"],
            "evaluation": evaluation,
            "notes": q.get("notes", ""),
        }
        results.append(full_entry)

        # Лог в файл
        with open(EVAL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(full_entry, ensure_ascii=False) + "\n")

        # Вывод в консоль
        status_icon = "✅" if evaluation["is_correct"] else "❌"
        print(f"       {status_icon} [{evaluation['status']}] {evaluation['explanation']}")
        print(f"       Время: {log_entry['duration_sec']}с | Длина: {log_entry['answer_length']} символов")
        print()

    # 4. Сводный отчёт
    print(f"\n{'=' * 60}")
    print("СВОДНЫЙ ОТЧЁТ")
    print(f"{'=' * 60}\n")

    total = len(results)
    correct = sum(1 for r in results if r["evaluation"]["is_correct"])
    tp = sum(1 for r in results if r["evaluation"]["status"] == "TP")
    tn = sum(1 for r in results if r["evaluation"]["status"] == "TN")
    fp = sum(1 for r in results if r["evaluation"]["status"] == "FP")
    fn = sum(1 for r in results if r["evaluation"]["status"] == "FN")

    print(f"  Всего вопросов:     {total}")
    print(f"  Корректных ответов: {correct}/{total} ({100*correct//total}%)")
    print()
    print(f"  Матрица результатов:")
    print(f"    TP (верный ответ):   {tp}")
    print(f"    TN (верный отказ):   {tn}")
    print(f"    FP (ложный ответ):   {fp}")
    print(f"    FN (ложный отказ):   {fn}")

    # Анализ пробелов
    print(f"\n{'─' * 60}")
    print("АНАЛИЗ ПРОБЕЛОВ\n")

    fn_results = [r for r in results if r["evaluation"]["status"] == "FN"]
    fp_results = [r for r in results if r["evaluation"]["status"] == "FP"]

    if fn_results:
        print("  Темы, по которым бот НЕ ОТВЕТИЛ (хотя должен был):")
        for r in fn_results:
            print(f"    • [{r['topic']}] {r['question']}")
            print(f"      Причина: {r['evaluation']['explanation']}")
            print(f"      Источники в индексе: {r['sources']}")
            print()
    else:
        print("  Ложных отказов (FN) нет — все ожидаемые ответы получены.\n")

    if fp_results:
        print("  Темы, по которым бот ОТВЕТИЛ (хотя не должен был):")
        for r in fp_results:
            print(f"    • [{r['topic']}] {r['question']}")
            print(f"      Ответ: {r['answer'][:100]}...")
            print()
    else:
        print("  Ложных ответов (FP) нет — все отказы корректны.\n")

    # Рекомендации
    print(f"{'─' * 60}")
    print("РЕКОМЕНДАЦИИ\n")

    deleted_topics = [r for r in fn_results if "УДАЛЁН" in r.get("notes", "")]
    other_fn = [r for r in fn_results if "УДАЛЁН" not in r.get("notes", "")]

    if deleted_topics:
        print("  Пробелы из-за удалённых файлов:")
        for r in deleted_topics:
            print(f"    → Восстановить данные по теме: {r['topic']}")

    if other_fn:
        print("  Пробелы в существующей базе:")
        for r in other_fn:
            print(f"    → Улучшить покрытие по теме: {r['topic']}")
            print(f"      Рекомендация: добавить/расширить документацию")

    if not fn_results and not fp_results:
        print("  База знаний полностью покрывает golden set. Рекомендуется:")
        print("    → Расширить набор вопросов")
        print("    → Добавить edge-case вопросы")

    print(f"\n  Лог сохранён: {os.path.abspath(EVAL_LOG_PATH)}")
    print(f"{'=' * 60}\n")


# === Main ===

if __name__ == "__main__":
    run_evaluation()
