"""Debot: Telegram-бот с DeepSeek/Qwen API и ограничителем запросов."""

from __future__ import annotations

import base64
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from telegram import ReplyKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
)
logger = logging.getLogger("debot")

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
AI_PROVIDER = os.getenv("AI_PROVIDER", "qwen").strip().lower()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip().rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash").strip()
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "").strip()
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "").strip().rstrip("/")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3-vl-plus").strip()
QWEN_FREE_QUOTA_ONLY_CONFIRMED = os.getenv("QWEN_FREE_QUOTA_ONLY_CONFIRMED", "false").lower() in ("1", "true", "yes")
QWEN_ALLOW_PAYG = os.getenv("QWEN_ALLOW_PAYG", "false").lower() in ("1", "true", "yes")
DEEPSEEK_ALLOW_PAID = os.getenv("DEEPSEEK_ALLOW_PAID", "false").lower() in ("1", "true", "yes")
MAX_API_CALLS_PER_DAY = int(os.getenv("MAX_API_CALLS_PER_DAY", "50"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "512"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "4000"))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_MB", "8")) * 1024 * 1024
USAGE_DB_PATH = Path(os.getenv("USAGE_DB_PATH", str(Path(__file__).with_name("debot_usage.sqlite3"))))
MAX_HISTORY_MESSAGES = 8
MAX_TELEGRAM_MESSAGE_LENGTH = 3900

MODES: dict[str, str] = {
    "Дебил 🤪": (
        "Ты комедийный персонаж-дурачок: отвечаешь нарочито глупо, иногда невпопад, "
        "путаешься в словах и можешь допускать безобидные ошибки. Это дружеская роль, "
        "не унижай пользователя и не выдавай опасную чушь за реальные инструкции."
    ),
    "Злой 😈": (
        "Ты ворчливый, раздражительный и саркастичный комедийный персонаж: огрызаешься "
        "и можешь слегка грубить, но остаёшься в рамках шутки, не угрожаешь и не травишь "
        "пользователя. На вопросы о безопасности отвечай точно."
    ),
    "Откровенный дебил 🤡": (
        "Ты абсурдный комедийный персонаж, который несёт нарочитую бессмыслицу и "
        "ломает логику ради шутки. Не превращай абсурд в опасный или вводящий в заблуждение "
        "совет: на медицинские, правовые, финансовые и связанные с безопасностью вопросы "
        "коротко предупреждай, что это роль, и отвечай достоверно."
    ),
    "Мемный 😂": (
        "Ты мемный саркастичный оптимист: даже нелепые ситуации описываешь так, будто "
        "всё шикарно, например называешь развалюху дворцом. Шутки доброжелательные, "
        "не оскорбляй пользователя и не искажай важные сведения о безопасности."
    ),
}

KEYBOARD = ReplyKeyboardMarkup(
    [["Дебил 🤪", "Злой 😈"], ["Откровенный дебил 🤡", "Мемный 😂"]],
    resize_keyboard=True,
)


def get_provider_config() -> tuple[str, str, str]:
    """Возвращает base URL, ключ и модель выбранного провайдера."""
    if AI_PROVIDER == "deepseek":
        base_url, api_key, model = DEEPSEEK_BASE_URL, DEEPSEEK_API_KEY, DEEPSEEK_MODEL
    elif AI_PROVIDER == "qwen":
        base_url, api_key, model = QWEN_BASE_URL, QWEN_API_KEY, QWEN_MODEL
    else:
        raise RuntimeError("AI_PROVIDER должен быть deepseek или qwen.")

    if AI_PROVIDER == "qwen" and not (QWEN_FREE_QUOTA_ONLY_CONFIRMED or QWEN_ALLOW_PAYG):
        raise RuntimeError(
            "Вызовы Qwen заблокированы, пока не включён режим Free Quota Only в консоли и не задано "
            "QWEN_FREE_QUOTA_ONLY_CONFIRMED=true. Для осознанного разрешения оплаты за пределами квоты "
            "вместо этого задай QWEN_ALLOW_PAYG=true."
        )
    if AI_PROVIDER == "deepseek" and not DEEPSEEK_ALLOW_PAID:
        raise RuntimeError(
            "DeepSeek тарифицируется по токенам. Чтобы явно разрешить платные вызовы, "
            "задай DEEPSEEK_ALLOW_PAID=true в .env."
        )
    if not api_key:
        variable = "DEEPSEEK_API_KEY" if AI_PROVIDER == "deepseek" else "QWEN_API_KEY"
        raise RuntimeError(f"Не задан {variable}. Добавь ключ API в локальный файл .env.")
    if not base_url:
        raise RuntimeError(
            "Не задан QWEN_BASE_URL. Скопируй OpenAI-compatible endpoint своего "
            "региона/Workspace из Alibaba Cloud Model Studio в .env."
        )
    if (
        AI_PROVIDER == "qwen"
        and QWEN_FREE_QUOTA_ONLY_CONFIRMED
        and "ap-southeast-1.maas.aliyuncs.com" not in base_url.lower()
    ):
        raise RuntimeError(
            "Флаг QWEN_FREE_QUOTA_ONLY_CONFIRMED разрешён только для Singapore endpoint. "
            "Для другого региона отключи флаг и включай оплату только осознанно через QWEN_ALLOW_PAYG=true."
        )
    if not model:
        raise RuntimeError("Имя AI-модели не должно быть пустым.")
    return base_url, api_key, model


def build_api_messages(
    mode: str,
    history: list[dict[str, str]],
    user_text: str,
    image_base64: str | None = None,
) -> list[dict[str, Any]]:
    """Строит запрос OpenAI-compatible chat completions для Qwen или DeepSeek."""
    selected_mode = mode if mode in MODES else "Дебил 🤪"
    recent_history = history[-MAX_HISTORY_MESSAGES:]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": MODES[selected_mode]},
        *recent_history,
    ]
    if image_base64:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": user_text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            },
        ]
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": user_text})
    return messages


def split_telegram_text(text: str, limit: int = MAX_TELEGRAM_MESSAGE_LENGTH) -> list[str]:
    """Делит длинный ответ на части, не теряя символы."""
    if not text:
        return ["Модель вернула пустой ответ. Попробуй отправить сообщение ещё раз."]
    return [text[index : index + limit] for index in range(0, len(text), limit)]


def reserve_daily_api_call(limit: int = MAX_API_CALLS_PER_DAY) -> tuple[bool, int]:
    """Атомарно резервирует один запрос в постоянном UTC-суточном лимите."""
    if limit < 1:
        return False, 0
    USAGE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).date().isoformat()
    with sqlite3.connect(USAGE_DB_PATH, timeout=10) as database:
        database.execute(
            "CREATE TABLE IF NOT EXISTS daily_usage (utc_day TEXT PRIMARY KEY, calls INTEGER NOT NULL)"
        )
        database.execute("BEGIN IMMEDIATE")
        row = database.execute(
            "SELECT calls FROM daily_usage WHERE utc_day = ?", (day,)
        ).fetchone()
        calls = row[0] if row else 0
        if calls >= limit:
            database.rollback()
            return False, calls
        calls += 1
        database.execute(
            "INSERT INTO daily_usage (utc_day, calls) VALUES (?, ?) "
            "ON CONFLICT(utc_day) DO UPDATE SET calls = excluded.calls",
            (day, calls),
        )
        database.commit()
        return True, calls


def get_daily_api_calls() -> int:
    if not USAGE_DB_PATH.exists():
        return 0
    day = datetime.now(timezone.utc).date().isoformat()
    with sqlite3.connect(USAGE_DB_PATH, timeout=10) as database:
        database.execute(
            "CREATE TABLE IF NOT EXISTS daily_usage (utc_day TEXT PRIMARY KEY, calls INTEGER NOT NULL)"
        )
        row = database.execute(
            "SELECT calls FROM daily_usage WHERE utc_day = ?", (day,)
        ).fetchone()
    return row[0] if row else 0


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, dict)]
        return "\n".join(part for part in parts if isinstance(part, str)).strip()
    return ""


async def call_ai(messages: list[dict[str, Any]]) -> str:
    """Отправляет мультимодальный запрос выбранному облачному API."""
    base_url, api_key, model = get_provider_config()
    timeout = httpx.Timeout(90.0, connect=10.0)
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.8,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
            )
    except httpx.ConnectError as exc:
        raise RuntimeError("Не удалось подключиться к AI API. Проверь endpoint и интернет.") from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError("AI API не ответил за 90 секунд. Попробуй ещё раз позже.") from exc
    except httpx.HTTPError as exc:
        logger.error("Ошибка HTTP-клиента API: %s", type(exc).__name__)
        raise RuntimeError("Ошибка соединения с AI API. Подробности есть в логе бота.") from exc

    if response.is_error:
        error_code = ""
        try:
            body = response.json()
            error = body.get("error", body) if isinstance(body, dict) else {}
            if isinstance(error, dict):
                error_code = str(error.get("code", ""))
        except ValueError:
            pass
        if "FreeTierOnly" in error_code:
            raise RuntimeError(
                "Бесплатная квота Qwen исчерпана; режим Free Quota Only остановил запрос, "
                "платный вызов не отправлен. Проверь остаток квоты в Model Studio."
            )
        if response.status_code in (401, 403):
            raise RuntimeError("API отклонил ключ или endpoint. Проверь ключ, регион и права модели.")
        if response.status_code == 429:
            raise RuntimeError(
                "API отклонил запрос по лимиту частоты/квоты. Подожди и проверь лимиты модели."
            )
        if response.status_code == 402 or "quota" in error_code.lower() or "balance" in error_code.lower():
            raise RuntimeError(
                "У API закончилась доступная квота или баланс. Проверь тариф и остаток квоты в консоли."
            )
        logger.warning("AI API вернул HTTP %s, error_code=%s", response.status_code, error_code or "unknown")
        raise RuntimeError(f"AI API вернул HTTP {response.status_code}. Проверь настройки провайдера.")

    try:
        body = response.json()
        answer = extract_text_content(body["choices"][0]["message"]["content"])
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("AI API вернул ответ в неожиданном формате.") from exc
    if not answer:
        raise RuntimeError("Модель вернула пустой ответ. Попробуй отправить сообщение ещё раз.")
    usage = body.get("usage", {})
    total_tokens = usage.get("total_tokens")
    if isinstance(total_tokens, int):
        logger.info("Provider=%s model=%s total_tokens=%s", AI_PROVIDER, model, total_tokens)
    return answer


async def send_long_reply(update: Update, text: str, *, with_keyboard: bool = False) -> None:
    message = update.effective_message
    if message is None:
        return
    chunks = split_telegram_text(text)
    for index, chunk in enumerate(chunks):
        markup = KEYBOARD if with_keyboard and index == len(chunks) - 1 else None
        await message.reply_text(chunk, reply_markup=markup)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["mode"] = "Дебил 🤪"
    context.user_data["history"] = []
    await send_long_reply(
        update,
        "Привет, я Дебот. Выбери режим кнопкой ниже — отвечаю через настроенный AI API.",
        with_keyboard=True,
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await send_long_reply(
        update,
        "Выбери режим кнопкой, отправь текст или фото.\n"
        "/start — начать заново и выбрать режим\n"
        "/reset — очистить историю диалога\n"
        "/status — показать провайдера и суточный лимит запросов\n"
        "Суточный предел общий для всех пользователей бота и хранится между перезапусками.",
        with_keyboard=True,
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["history"] = []
    await send_long_reply(update, "Историю диалога очистил. Режим оставил прежним.", with_keyboard=True)


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    calls = get_daily_api_calls()
    try:
        _, api_key, model = get_provider_config()
        configured = "ключ задан"
    except RuntimeError as exc:
        configured = str(exc)
        model = "не настроена"
    text = (
        f"Провайдер: {AI_PROVIDER}; модель: {model}; {configured}.\n"
        f"Сегодня использовано API-запросов (UTC): {calls}/{MAX_API_CALLS_PER_DAY}.\n"
        "Команда /status не отправляет запрос модели и не расходует токены."
    )
    await send_long_reply(update, text, with_keyboard=True)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return

    if message.text in MODES:
        context.user_data["mode"] = message.text
        context.user_data["history"] = []
        await send_long_reply(
            update,
            f"Режим «{message.text}» активирован. Историю очистил — присылай текст или фото.",
            with_keyboard=True,
        )
        return

    image_base64: str | None = None
    if message.photo:
        photo = message.photo[-1]
        if photo.file_size and photo.file_size > MAX_IMAGE_BYTES:
            await send_long_reply(
                update,
                f"Фото слишком большое (лимит {MAX_IMAGE_BYTES // (1024 * 1024)} МБ). Пришли фото поменьше.",
                with_keyboard=True,
            )
            return
        try:
            telegram_file = await context.bot.get_file(photo.file_id)
            image_bytes = await telegram_file.download_as_bytearray()
        except Exception:
            logger.exception("Не удалось скачать фото из Telegram")
            await send_long_reply(update, "Не получилось загрузить фото из Telegram. Попробуй отправить его ещё раз.")
            return
        if len(image_bytes) > MAX_IMAGE_BYTES:
            await send_long_reply(
                update,
                f"Фото слишком большое (лимит {MAX_IMAGE_BYTES // (1024 * 1024)} МБ). Пришли фото поменьше.",
                with_keyboard=True,
            )
            return
        image_base64 = base64.b64encode(image_bytes).decode("ascii")
        user_text = message.caption or "Опиши, что изображено на фото, и ответь с учётом выбранного режима."
    elif message.text:
        user_text = message.text.strip()
        if not user_text:
            return
    else:
        await send_long_reply(update, "Пока поддерживаются текст и фотографии.", with_keyboard=True)
        return

    if len(user_text) > MAX_INPUT_CHARS:
        await send_long_reply(
            update,
            f"Сообщение слишком длинное. Лимит — {MAX_INPUT_CHARS} символов, чтобы ограничить расход токенов.",
            with_keyboard=True,
        )
        return

    try:
        get_provider_config()
    except RuntimeError as exc:
        await send_long_reply(update, str(exc), with_keyboard=True)
        return

    allowed, calls_today = reserve_daily_api_call()
    if not allowed:
        await send_long_reply(
            update,
            f"Дневной лимит API-запросов достигнут ({calls_today}/{MAX_API_CALLS_PER_DAY}, UTC). "
            "Новые запросы остановлены до следующего дня.",
            with_keyboard=True,
        )
        return

    mode = context.user_data.get("mode", "Дебил 🤪")
    history = context.user_data.get("history", [])
    api_messages = build_api_messages(mode, history, user_text, image_base64)
    try:
        await context.bot.send_chat_action(chat_id=message.chat_id, action=ChatAction.TYPING)
        answer = await call_ai(api_messages)
    except RuntimeError as exc:
        logger.warning("AI-запрос не выполнен: %s", exc)
        await send_long_reply(update, str(exc), with_keyboard=True)
        return
    except Exception:
        logger.exception("Непредвиденная ошибка во время AI-запроса")
        await send_long_reply(update, "Внутренняя ошибка бота. Запрос учтён в дневном лимите; проверь логи.")
        return

    updated_history = [
        *history,
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": answer},
    ]
    context.user_data["history"] = updated_history[-MAX_HISTORY_MESSAGES:]
    await send_long_reply(update, answer, with_keyboard=True)


def build_application() -> Application:
    if not BOT_TOKEN:
        raise RuntimeError("Не задан BOT_TOKEN. Укажи новый токен Telegram в локальном файле .env.")
    if MAX_API_CALLS_PER_DAY < 1:
        raise RuntimeError("MAX_API_CALLS_PER_DAY должен быть не меньше 1.")
    if MAX_OUTPUT_TOKENS < 1 or MAX_INPUT_CHARS < 1:
        raise RuntimeError("Лимиты входа и выхода должны быть положительными числами.")
    get_provider_config()

    application = ApplicationBuilder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(
        MessageHandler((filters.TEXT & ~filters.COMMAND) | filters.PHOTO, handle_message)
    )
    return application


if __name__ == "__main__":
    build_application().run_polling(allowed_updates=Update.ALL_TYPES)
