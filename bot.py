"""Debot: Telegram-бот с полностью локальной генерацией через Ollama."""

from __future__ import annotations

import base64
import logging
import os
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
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b").strip()
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_MB", "15")) * 1024 * 1024
MAX_HISTORY_MESSAGES = 12
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


def build_ollama_messages(
    mode: str,
    history: list[dict[str, str]],
    user_text: str,
    image_base64: str | None = None,
) -> list[dict[str, Any]]:
    """Формирует совместимый с Ollama /api/chat запрос."""
    selected_mode = mode if mode in MODES else "Дебил 🤪"
    recent_history = history[-MAX_HISTORY_MESSAGES:]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": MODES[selected_mode]},
        *recent_history,
    ]
    user_message: dict[str, Any] = {"role": "user", "content": user_text}
    if image_base64:
        user_message["images"] = [image_base64]
    messages.append(user_message)
    return messages


def split_telegram_text(text: str, limit: int = MAX_TELEGRAM_MESSAGE_LENGTH) -> list[str]:
    """Делит длинный ответ на части, не теряя символы."""
    if not text:
        return ["Модель вернула пустой ответ. Попробуй отправить сообщение ещё раз."]
    return [text[index : index + limit] for index in range(0, len(text), limit)]


async def ollama_chat(messages: list[dict[str, Any]]) -> str:
    """Отправляет запрос локальному Ollama-серверу, а не платному AI API."""
    timeout = httpx.Timeout(OLLAMA_TIMEOUT_SECONDS, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.8},
                },
            )
            if response.status_code == 404:
                raise RuntimeError(
                    f"Ollama не нашла модель {OLLAMA_MODEL}. Установи её командой: "
                    f"ollama pull {OLLAMA_MODEL}"
                )
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Не удалось подключиться к Ollama по адресу {OLLAMA_BASE_URL}. "
                "Запусти Ollama и проверь OLLAMA_BASE_URL в .env."
            ) from exc
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                "Локальная модель не успела ответить. Попробуй ещё раз или увеличь "
                "OLLAMA_TIMEOUT_SECONDS в .env."
            ) from exc
        except httpx.HTTPStatusError as exc:
            logger.error("Ollama вернула HTTP %s", exc.response.status_code)
            raise RuntimeError(
                f"Ollama вернула HTTP {exc.response.status_code}. "
                "Проверь состояние локального сервера и установленную модель."
            ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("Ollama вернула ответ в неожиданном формате.") from exc
    content = payload.get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Локальная модель вернула пустой ответ; попробуй ещё раз.")
    return content.strip()


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
        "Привет, я Дебот. Выбери режим кнопкой ниже — отвечаю локальной моделью на твоём устройстве.",
        with_keyboard=True,
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await send_long_reply(
        update,
        "Выбери режим кнопкой, отправь текст или фото.\n"
        "/start — начать заново и выбрать режим\n"
        "/reset — очистить историю диалога\n"
        "/status — проверить связь с локальной моделью\n"
        "Генерация выполняется на локальной Ollama-модели; платный AI API не используется.",
        with_keyboard=True,
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["history"] = []
    await send_long_reply(update, "Историю диалога очистил. Режим оставил прежним.", with_keyboard=True)


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    timeout = httpx.Timeout(5.0, connect=3.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
        data = response.json()
        installed = {item.get("name") or item.get("model") for item in data.get("models", [])}
        if OLLAMA_MODEL in installed:
            text = f"Ollama доступна. Модель {OLLAMA_MODEL} установлена; генерация локальная."
        else:
            text = (
                f"Ollama доступна, но модель {OLLAMA_MODEL} не установлена. "
                f"Выполни: ollama pull {OLLAMA_MODEL}"
            )
    except (httpx.HTTPError, ValueError) as exc:
        logger.info("Проверка Ollama не удалась: %s", type(exc).__name__)
        text = (
            f"Ollama недоступна по адресу {OLLAMA_BASE_URL}. "
            "Запусти её и проверь настройки в .env."
        )
    await send_long_reply(update, text, with_keyboard=True)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return

    # Нажатие кнопки режима не отправляется модели как обычное сообщение.
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
                f"Фото слишком большое (лимит {MAX_IMAGE_BYTES // (1024 * 1024)} МБ). Пришли файл поменьше.",
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
                f"Фото слишком большое (лимит {MAX_IMAGE_BYTES // (1024 * 1024)} МБ). Пришли файл поменьше.",
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

    mode = context.user_data.get("mode", "Дебил 🤪")
    history = context.user_data.get("history", [])
    messages = build_ollama_messages(mode, history, user_text, image_base64)

    try:
        await context.bot.send_chat_action(chat_id=message.chat_id, action=ChatAction.TYPING)
        answer = await ollama_chat(messages)
    except RuntimeError as exc:
        logger.warning("Локальная генерация не удалась: %s", exc)
        await send_long_reply(update, str(exc), with_keyboard=True)
        return
    except Exception:
        logger.exception("Непредвиденная ошибка при генерации ответа")
        await send_long_reply(update, "Произошла внутренняя ошибка при обращении к локальной модели. Проверь логи бота.")
        return

    updated_history = [*history, {"role": "user", "content": user_text}, {"role": "assistant", "content": answer}]
    context.user_data["history"] = updated_history[-MAX_HISTORY_MESSAGES:]
    await send_long_reply(update, answer, with_keyboard=True)


def build_application() -> Application:
    if not BOT_TOKEN:
        raise RuntimeError(
            "Не задан BOT_TOKEN. Создай файл .env по примеру .env.example "
            "и укажи новый токен Telegram-бота."
        )
    if not OLLAMA_MODEL:
        raise RuntimeError("Переменная OLLAMA_MODEL не должна быть пустой.")

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
