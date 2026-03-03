"""
transcribe.py - Автоматична транскрипція аудіо файлів (українська мова)
Спочатку використовує Groq API (whisper-large-v3), при помилці — локальний faster-whisper.

Налаштування - у файлі config.env поруч з цим скриптом.

Структура кожної папки:
  recordings/
    audio.mp3            <- вхідні файли
    2026-02-27.md        <- результат транскрипції
    archive/
      2026-02-27/
        audio.mp3
"""

import shutil
from datetime import datetime
from pathlib import Path

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Не знайдено faster-whisper. Встановіть: pip install faster-whisper")
    exit(1)

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


AUDIO_EXTENSIONS = ("*.mp3", "*.wav", "*.m4a", "*.ogg", "*.flac", "*.aac")

# Groq має ліміт 25 МБ на файл
GROQ_MAX_FILE_SIZE_MB = 25


# --- КОНФІГ ------------------------------------------------------------------

def load_config() -> dict:
    config_path = Path(__file__).parent / "config.env"
    if not config_path.exists():
        print(f"ПОМИЛКА: Не знайдено config.env у {config_path.parent}")
        exit(1)

    config = {
        "RECORDINGS_DIRS": [],
        "MODEL_SIZE": "medium",
        "LANGUAGE": "uk",
        "GROQ_API": "",
    }
    current_key = None

    with open(config_path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if "=" in line:
                key, _, value = line.partition("=")
                current_key = key.strip()
                value = value.strip()
                if current_key == "RECORDINGS_DIRS":
                    if value:
                        config["RECORDINGS_DIRS"].append(Path(value))
                else:
                    config[current_key] = value
            elif current_key == "RECORDINGS_DIRS" and line:
                config["RECORDINGS_DIRS"].append(Path(line))

    if not config["RECORDINGS_DIRS"]:
        print("ПОМИЛКА: У config.env не вказано жодної папки в RECORDINGS_DIRS")
        exit(1)

    return config


# --- ДОПОМІЖНІ ФУНКЦІЇ -------------------------------------------------------

def get_today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def get_md_path(base_dir: Path, date_str: str) -> Path:
    return base_dir / f"{date_str}.md"


def get_archive_dir(base_dir: Path, date_str: str) -> Path:
    return base_dir / "archive" / date_str


def append_to_md(md_path: Path, filename: str, text: str, date_str: str):
    is_new = not md_path.exists()
    with open(md_path, "a", encoding="utf-8") as f:
        if is_new:
            f.write(f"# Транскрипції {date_str}\n\n")
        title = Path(filename).stem
        f.write(f"## {title}\n\n")
        f.write(text.strip())
        f.write("\n\n---\n\n")
    action = "створено" if is_new else "оновлено"
    print(f"  MD файл {action}: {md_path.name}")


def move_to_archive(audio_path: Path, archive_dir: Path):
    archive_dir.mkdir(parents=True, exist_ok=True)
    dest = archive_dir / audio_path.name
    if dest.exists():
        suffix = datetime.now().strftime("%H%M%S")
        dest = archive_dir / f"{audio_path.stem}_{suffix}{audio_path.suffix}"
    shutil.move(str(audio_path), str(dest))
    print(f"  Архівовано: archive/{archive_dir.name}/{dest.name}")


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# --- ТРАНСКРИПЦІЯ ЧЕРЕЗ GROQ -------------------------------------------------

def transcribe_with_groq(groq_client: Groq, audio_path: Path, language: str) -> str:
    """Транскрипція через Groq API (whisper-large-v3). Повертає текст або кидає виняток."""
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    if file_size_mb > GROQ_MAX_FILE_SIZE_MB:
        raise ValueError(f"Файл {file_size_mb:.1f} МБ перевищує ліміт Groq {GROQ_MAX_FILE_SIZE_MB} МБ")

    print(f"  [Groq] Транскрипція: {audio_path.name} ({file_size_mb:.1f} МБ)...")
    # Groq вимагає розширення у нижньому регістрі (напр. .MP3 → .mp3)
    normalized_name = audio_path.stem + audio_path.suffix.lower()

    with open(audio_path, "rb") as f:
        result = groq_client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=(normalized_name, f),
            language=language,
            response_format="text",
        )
    # При response_format="text" повертається рядок напряму
    text = result if isinstance(result, str) else result.text
    print(f"  [Groq] Готово (слів: {len(text.split())})")
    return text


# --- ТРАНСКРИПЦІЯ ЛОКАЛЬНО ---------------------------------------------------

def transcribe_locally(model: WhisperModel, audio_path: Path, language: str) -> str:
    """Транскрипція локальною faster-whisper моделлю."""
    print(f"  [Локально] Транскрипція: {audio_path.name} ...")
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    duration = info.duration
    show_progress = duration > 120
    next_checkpoint = 30.0

    collected = []
    for segment in segments:
        text = segment.text.strip()
        collected.append(text)
        if show_progress and segment.start >= next_checkpoint:
            print(f"    [{format_time(segment.start)} / {format_time(duration)}] ...")
            next_checkpoint += 30.0

    full_text = " ".join(collected)
    print(f"  [Локально] Готово (мова: {info.language}, слів: {len(full_text.split())})")
    return full_text


# --- ТРАНСКРИПЦІЯ З FALLBACK -------------------------------------------------

def transcribe_file(
    audio_path: Path,
    language: str,
    groq_client: "Groq | None",
    local_model: "WhisperModel | None",
    model_size: str,
) -> str:
    """
    Спочатку пробує Groq. При будь-якій помилці — падає на локальну модель.
    Локальна модель завантажується ліниво (лише якщо знадобиться).
    """
    if groq_client is not None:
        try:
            return transcribe_with_groq(groq_client, audio_path, language)
        except Exception as e:
            print(f"  [Groq] Помилка: {e}")
            print(f"  [Groq] Перемикаємось на локальну модель...")

    # Ліниве завантаження локальної моделі
    if local_model[0] is None:
        print(f"\n  Завантаження локальної моделі '{model_size}'...")
        local_model[0] = WhisperModel(model_size, device="cpu", compute_type="int8")

    return transcribe_locally(local_model[0], audio_path, language)


# --- ГОЛОВНА ЛОГІКА ----------------------------------------------------------

def main():
    config = load_config()
    dirs = config["RECORDINGS_DIRS"]
    model_size = config["MODEL_SIZE"]
    language = config["LANGUAGE"]
    groq_api_key = config.get("GROQ_API", "").strip()

    # Ініціалізація Groq клієнта
    groq_client = None
    if groq_api_key and GROQ_AVAILABLE:
        groq_client = Groq(api_key=groq_api_key)
        print("Groq API: увімкнено (whisper-large-v3)")
    elif not GROQ_AVAILABLE and groq_api_key:
        print("⚠ Groq API ключ знайдено, але бібліотека не встановлена.")
        print("  Встановіть: pip install groq")
        print("  Буде використано локальну модель.")
    else:
        print("Groq API: не налаштовано — використовуємо локальну модель")

    # Збір файлів
    tasks = []
    for recordings_dir in dirs:
        if not recordings_dir.exists():
            print(f"[!] Папка не знайдена, пропускаємо: {recordings_dir}")
            continue
        files = sorted(f for ext in AUDIO_EXTENSIONS for f in recordings_dir.glob(ext))
        for f in files:
            tasks.append((recordings_dir, f))

    if not tasks:
        print("Аудіо файлів не знайдено у жодній з папок.")
        return

    print(f"Знайдено {len(tasks)} файл(ів) у {len(dirs)} папці(ах)\n")

    # local_model передається як список з одним елементом для лінивої ініціалізації
    local_model = [None]

    date_str = get_today_str()
    success_count = 0
    error_count = 0

    for i, (recordings_dir, audio_path) in enumerate(tasks):
        print(f"[{i+1}/{len(tasks)}] {audio_path.name}  ({recordings_dir})")
        try:
            text = transcribe_file(
                audio_path, language, groq_client, local_model, model_size
            )
            md_path = get_md_path(recordings_dir, date_str)
            archive_dir = get_archive_dir(recordings_dir, date_str)
            append_to_md(md_path, audio_path.name, text, date_str)
            move_to_archive(audio_path, archive_dir)
            success_count += 1
        except Exception as e:
            print(f"  Помилка: {e}")
            error_count += 1

    print(f"\n{'='*50}")
    print(f"Оброблено: {success_count} | Помилок: {error_count}")


if __name__ == "__main__":
    main()
