"""
prepare_data.py — Замена терминов Star Wars на корпоративный лор «МегаОфис».

Читает статьи из data/source/, применяет словарь замен terms_map.json,
сохраняет результат в data/knowledge_base/.

Использование:
    python scripts/prepare_data.py

Алгоритм замены (двухпроходный, через плейсхолдеры):
  1. Сортируем термины по длине (от длинных к коротким).
  2. Первый проход: заменяем каждый термин на уникальный плейсхолдер <<TERM_NNN>>.
     Это предотвращает каскадные замены ("Хан" внутри "Ханов").
  3. Второй проход: заменяем плейсхолдеры на финальные значения.
"""

import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..")
SOURCE_DIR = os.path.join(ROOT_DIR, "data", "source")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "knowledge_base")
TERMS_MAP_PATH = os.path.join(ROOT_DIR, "terms_map.json")


def load_terms_map(path: str) -> dict[str, str]:
    """
    Загружает terms_map.json и собирает плоский словарь замен.
    Пропускает ключ _meta.
    Сортирует по длине ключа (от длинных к коротким),
    чтобы "Энакин Скайуокер" заменялся раньше, чем "Энакин".
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    flat = {}
    for category, mappings in raw.items():
        if category == "_meta":
            continue
        flat.update(mappings)

    sorted_terms = dict(
        sorted(flat.items(), key=lambda x: len(x[0]), reverse=True)
    )
    return sorted_terms


def apply_replacements(text: str, terms: dict[str, str]) -> tuple[str, int]:
    """
    Двухпроходная замена через плейсхолдеры.
    Предотвращает каскадные замены ("Хан" внутри уже заменённого "Ханов").
    Возвращает (изменённый текст, количество замен).
    """
    placeholders: dict[str, str] = {}
    total_replacements = 0

    # Первый проход: термины → плейсхолдеры
    for i, (original, replacement) in enumerate(terms.items()):
        placeholder = f"\x00TERM_{i:04d}\x00"
        count = text.count(original)
        if count > 0:
            text = text.replace(original, placeholder)
            placeholders[placeholder] = replacement
            total_replacements += count

    # Второй проход: плейсхолдеры → финальные значения
    for placeholder, replacement in placeholders.items():
        text = text.replace(placeholder, replacement)

    return text, total_replacements


def process_files(source_dir: str, output_dir: str, terms: dict[str, str]):
    """Обрабатывает все файлы из source_dir, сохраняет в output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(source_dir) if f.endswith((".md", ".txt"))]
    if not files:
        print(f"⚠ Нет файлов в {os.path.abspath(source_dir)}")
        print("  Сначала запустите: python scripts/download_articles.py")
        return

    print(f"Обработка {len(files)} файлов...")
    print(f"  Источник:  {os.path.abspath(source_dir)}")
    print(f"  Результат: {os.path.abspath(output_dir)}")
    print(f"  Терминов для замены: {len(terms)}\n")

    total_replacements = 0

    for filename in sorted(files):
        source_path = os.path.join(source_dir, filename)
        output_path = os.path.join(output_dir, filename)

        with open(source_path, "r", encoding="utf-8") as f:
            original_text = f.read()

        modified_text, replacements = apply_replacements(original_text, terms)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(modified_text)

        status = f"({replacements} замен)" if replacements > 0 else "(без замен)"
        print(f"  ✓ {filename} {status}")
        total_replacements += replacements

    print(f"\n{'='*50}")
    print(f"Готово! Обработано файлов: {len(files)}")
    print(f"Всего замен: {total_replacements}")
    print(f"Результат в: {os.path.abspath(output_dir)}")


def main():
    if not os.path.exists(TERMS_MAP_PATH):
        print(f"✗ Не найден файл словаря: {TERMS_MAP_PATH}")
        return

    if not os.path.exists(SOURCE_DIR):
        print(f"✗ Не найдена папка с исходными файлами: {SOURCE_DIR}")
        print("  Сначала запустите: python scripts/download_articles.py")
        return

    terms = load_terms_map(TERMS_MAP_PATH)
    print(f"Загружен словарь: {len(terms)} терминов\n")

    process_files(SOURCE_DIR, OUTPUT_DIR, terms)


if __name__ == "__main__":
    main()
