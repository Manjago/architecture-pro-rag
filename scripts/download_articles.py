"""
download_articles.py — Скачивание статей из русской Wikipedia по вселенной Star Wars.

Использование:
    python scripts/download_articles.py

Результат:
    Папка data/source/ с .md файлами (одна статья = один файл).
"""

import os
import time
import wikipedia

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "source")

# Список статей для скачивания.
# Каждый элемент — точное название статьи в русской Wikipedia.
# Подобраны так, чтобы покрыть: персонажей, локации, технологии,
# организации, события, корабли — всё, что есть в terms_map.json.
ARTICLES = [
    # === Персонажи (15) ===
    "Люк Скайуокер",
    "Дарт Вейдер",
    "Лея Органа",
    "Хан Соло",
    "Оби-Ван Кеноби",
    "Йода",
    "Палпатин",
    "Чубакка",
    "Боба Фетт",
    "Падме Амидала",
    "Мейс Винду",
    "Дарт Мол",
    "Граф Дуку",
    "Квай-Гон Джинн",
    "Лэндо Калриссиан",
    # === Локации и планеты (7) ===
    "Звезда Смерти",
    "Татуин",
    "Корусант",
    "Набу (Звёздные войны)",
    "Хот (Звёздные войны)",
    "Эндор (Звёздные войны)",
    "Мустафар",
    # === Организации (4) ===
    "Галактическая Империя (Звёздные войны)",
    "Джедай",
    "Ситхи",
    "Альянс повстанцев",
    # === Технологии и концепции (5) ===
    "Световой меч",
    "Сила (Звёздные войны)",
    "Звёздные войны",
    "Войны клонов (Звёздные войны)",
    "Гиперпространство (Звёздные войны)",
    # === Корабли (3) ===
    "Тысячелетний сокол",
    "Звёздный разрушитель типа «Имперский»",
    "X-wing",
    # === Фильмы / события (4) ===
    "Звёздные войны. Эпизод IV: Новая надежда",
    "Звёздные войны. Эпизод V: Империя наносит ответный удар",
    "Звёздные войны. Эпизод VI: Возвращение джедая",
    "Звёздные войны. Эпизод III: Месть ситхов",
]


def sanitize_filename(title: str) -> str:
    """Превращает название статьи в безопасное имя файла."""
    name = title.replace(" ", "_")
    # Убираем скобки и спецсимволы
    for ch in ['(', ')', '«', '»', ':', '"', '/', '\\', '?', '*']:
        name = name.replace(ch, "")
    return name.strip("_") + ".md"


def download_article(title: str) -> str | None:
    """Скачивает текст статьи. Возвращает None при ошибке."""
    try:
        page = wikipedia.page(title, auto_suggest=False)
        return page.content
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"  ⚠ Неоднозначность для '{title}', варианты: {e.options[:3]}")
        # Пробуем первый вариант
        try:
            page = wikipedia.page(e.options[0], auto_suggest=False)
            return page.content
        except Exception:
            return None
    except wikipedia.exceptions.PageError:
        print(f"  ✗ Страница не найдена: '{title}'")
        return None
    except Exception as e:
        print(f"  ✗ Ошибка при загрузке '{title}': {e}")
        return None


def main():
    wikipedia.set_lang("ru")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Скачиваем {len(ARTICLES)} статей из русской Wikipedia...")
    print(f"Папка назначения: {os.path.abspath(OUTPUT_DIR)}\n")

    success = 0
    errors = 0

    for i, title in enumerate(ARTICLES, 1):
        filename = sanitize_filename(title)
        filepath = os.path.join(OUTPUT_DIR, filename)

        # Пропускаем уже скачанные
        if os.path.exists(filepath):
            print(f"  [{i}/{len(ARTICLES)}] ✓ Уже есть: {filename}")
            success += 1
            continue

        print(f"  [{i}/{len(ARTICLES)}] Скачиваю: {title}...", end=" ")
        content = download_article(title)

        if content:
            # Добавляем заголовок в начало файла
            md_content = f"# {title}\n\n{content}"
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(md_content)
            print(f"✓ ({len(content)} символов)")
            success += 1
        else:
            errors += 1

        # Пауза между запросами, чтобы не нагружать API
        time.sleep(0.5)

    print(f"\n{'='*50}")
    print(f"Готово! Скачано: {success}, ошибок: {errors}")
    print(f"Файлы в: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
