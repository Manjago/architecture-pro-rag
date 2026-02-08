"""
telegram_bot.py — Telegram-интерфейс для RAG-бота.

Тариф Про: чат-бот в Телеграме.

Использует тот же RAG-пайплайн, что и консольный rag_bot.py:
  запрос → эмбеддинг → FAISS → фильтрация → промпт → Ollama → ответ.

Использование:
    # 1. Создать бота через @BotFather в Telegram, получить токен
    # 2. Установить зависимости:
    pip install python-telegram-bot

    # 3. Запустить (Ollama должна работать):
    TELEGRAM_BOT_TOKEN=your_token python scripts/telegram_bot.py

    # Или через .env файл:
    echo "TELEGRAM_BOT_TOKEN=your_token" > .env
    python scripts/telegram_bot.py

Зависимости:
    pip install python-telegram-bot langchain langchain-community langchain-ollama faiss-cpu sentence-transformers
"""

import logging
import os
import sys

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Импортируем RAG-компоненты из существующих скриптов
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_index import E5Embeddings, EMBEDDING_MODEL
from rag_bot import (
    INDEX_DIR,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_NUM_CTX,
    ask,
    load_index,
    load_llm,
)

# === Настройки ===

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)


# === Загрузка токена ===

def get_token() -> str:
    """Получает токен бота из переменной окружения или .env файла."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")

    if not token:
        # Пробуем .env файл
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("TELEGRAM_BOT_TOKEN="):
                        token = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break

    if not token:
        print("❌ Не найден TELEGRAM_BOT_TOKEN!")
        print()
        print("Варианты:")
        print("  1. Переменная окружения:")
        print("     TELEGRAM_BOT_TOKEN=your_token python scripts/telegram_bot.py")
        print()
        print("  2. Файл .env в корне проекта:")
        print('     echo "TELEGRAM_BOT_TOKEN=your_token" > .env')
        print()
        print("Получить токен: @BotFather в Telegram → /newbot")
        sys.exit(1)

    return token


# === Хэндлеры ===

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start."""
    await update.message.reply_text(
        "👋 Привет! Я RAG-бот корпоративной базы знаний «МегаОфис».\n\n"
        "Задайте мне вопрос по документации, и я постараюсь найти ответ.\n\n"
        "Команды:\n"
        "/start — это сообщение\n"
        "/help — справка\n\n"
        "Просто напишите вопрос — и я отвечу!"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help."""
    await update.message.reply_text(
        "📚 *RAG-бот «МегаОфис»*\n\n"
        "Я ищу ответы в корпоративной базе знаний.\n"
        "Если информации нет — честно скажу об этом.\n\n"
        "Примеры вопросов:\n"
        "• Кто такой Лёша Облаков?\n"
        "• Что такое Офис Гибели?\n"
        "• Чем известен Agile-Коуч Йодин?\n\n"
        "⚙️ Режим защиты: full (pre-prompt + post-фильтрация)\n"
        "🤖 Модель: llama3.1 (Ollama, локально)\n"
        "📊 Индекс: FAISS + multilingual-e5-large",
        parse_mode="Markdown",
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений — основной RAG-пайплайн."""
    question = update.message.text.strip()

    if not question:
        return

    logger.info(f"Вопрос от {update.effective_user.username}: {question}")

    # Отправляем «печатает...»
    await update.message.chat.send_action("typing")

    try:
        # Вызываем RAG-пайплайн (режим full — с защитой)
        vectorstore = context.bot_data["vectorstore"]
        llm = context.bot_data["llm"]

        answer = ask(
            question=question,
            vectorstore=vectorstore,
            llm=llm,
            filter_mode="full",
            verbose=False,
        )

        # Ограничиваем длину ответа (Telegram limit = 4096 символов)
        if len(answer) > 4000:
            answer = answer[:4000] + "\n\n... (ответ обрезан)"

        await update.message.reply_text(answer)
        logger.info(f"Ответ отправлен ({len(answer)} символов)")

    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        await update.message.reply_text(
            "⚠️ Произошла ошибка при обработке запроса. "
            "Убедитесь, что Ollama запущена и попробуйте ещё раз."
        )


# === Main ===

def main():
    """Запуск Telegram-бота."""
    print("=" * 60)
    print("RAG-бот «МегаОфис» — Telegram-интерфейс")
    print("=" * 60)

    # 1. Токен
    token = get_token()
    print(f"  ✅ Токен найден")

    # 2. Загружаем RAG-компоненты
    print("\n  Загрузка RAG-пайплайна...")
    vectorstore = load_index()
    llm = load_llm()

    # 3. Создаём приложение
    app = Application.builder().token(token).build()

    # Сохраняем vectorstore и llm в bot_data для доступа из хэндлеров
    app.bot_data["vectorstore"] = vectorstore
    app.bot_data["llm"] = llm

    # 4. Регистрируем хэндлеры
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # 5. Запускаем polling
    print("\n  🚀 Бот запущен! Ожидаю сообщения...")
    print("  Ctrl+C для остановки\n")

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
