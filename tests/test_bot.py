import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bot


class ProviderMessageTests(unittest.TestCase):
    def test_selected_persona_is_sent_as_system_message(self):
        messages = bot.build_api_messages("Мемный 😂", [], "Привет")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], bot.MODES["Мемный 😂"])
        self.assertEqual(messages[-1], {"role": "user", "content": "Привет"})

    def test_unknown_mode_uses_default_persona(self):
        messages = bot.build_api_messages("неизвестно", [], "Тест")
        self.assertEqual(messages[0]["content"], bot.MODES["Дебил 🤪"])

    def test_image_is_encoded_in_openai_compatible_content(self):
        messages = bot.build_api_messages("Злой 😈", [], "Что на фото?", "aGVsbG8=")
        self.assertEqual(messages[-1]["role"], "user")
        blocks = messages[-1]["content"]
        self.assertEqual(blocks[0], {"type": "text", "text": "Что на фото?"})
        self.assertEqual(blocks[1]["image_url"]["url"], "data:image/jpeg;base64,aGVsbG8=")

    def test_history_is_limited_to_recent_messages(self):
        history = [{"role": "user", "content": str(i)} for i in range(bot.MAX_HISTORY_MESSAGES + 4)]
        messages = bot.build_api_messages("Дебил 🤪", history, "новое")
        self.assertEqual(len(messages), bot.MAX_HISTORY_MESSAGES + 2)
        self.assertEqual(messages[1]["content"], "4")

    def test_qwen_requires_explicit_free_quota_or_paid_consent(self):
        common = (
            patch.object(bot, "AI_PROVIDER", "qwen"),
            patch.object(bot, "QWEN_API_KEY", "test-key"),
            patch.object(bot, "QWEN_BASE_URL", "https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"),
        )
        with common[0], common[1], common[2], patch.object(bot, "QWEN_FREE_QUOTA_ONLY_CONFIRMED", False), patch.object(
            bot, "QWEN_ALLOW_PAYG", False
        ):
            with self.assertRaisesRegex(RuntimeError, "QWEN_FREE_QUOTA_ONLY_CONFIRMED"):
                bot.get_provider_config()
        with common[0], common[1], common[2], patch.object(bot, "QWEN_FREE_QUOTA_ONLY_CONFIRMED", True), patch.object(
            bot, "QWEN_ALLOW_PAYG", False
        ), patch.object(bot, "QWEN_MODEL", "qwen3-vl-plus"):
            self.assertEqual(
                bot.get_provider_config(),
                ("https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1", "test-key", "qwen3-vl-plus"),
            )

    def test_free_quota_confirmation_rejects_non_singapore_endpoint(self):
        with patch.object(bot, "AI_PROVIDER", "qwen"), patch.object(
            bot, "QWEN_API_KEY", "test-key"
        ), patch.object(bot, "QWEN_BASE_URL", "https://workspace.us-east-1.maas.aliyuncs.com/compatible-mode/v1"), patch.object(
            bot, "QWEN_FREE_QUOTA_ONLY_CONFIRMED", True
        ), patch.object(bot, "QWEN_ALLOW_PAYG", False):
            with self.assertRaisesRegex(RuntimeError, "Singapore endpoint"):
                bot.get_provider_config()

    def test_deepseek_requires_explicit_paid_consent(self):
        with patch.object(bot, "AI_PROVIDER", "deepseek"), patch.object(
            bot, "DEEPSEEK_API_KEY", "test-key"
        ), patch.object(bot, "DEEPSEEK_ALLOW_PAID", False):
            with self.assertRaisesRegex(RuntimeError, "DEEPSEEK_ALLOW_PAID=true"):
                bot.get_provider_config()


class UsageLimitTests(unittest.TestCase):
    def test_daily_limit_persists_and_stops_new_calls(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            bot, "USAGE_DB_PATH", Path(temp_dir) / "usage.sqlite3"
        ):
            self.assertEqual(bot.reserve_daily_api_call(limit=2), (True, 1))
            self.assertEqual(bot.reserve_daily_api_call(limit=2), (True, 2))
            self.assertEqual(bot.reserve_daily_api_call(limit=2), (False, 2))
            self.assertEqual(bot.get_daily_api_calls(), 2)

    def test_zero_limit_blocks_calls(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            bot, "USAGE_DB_PATH", Path(temp_dir) / "usage.sqlite3"
        ):
            self.assertEqual(bot.reserve_daily_api_call(limit=0), (False, 0))


class OutputTests(unittest.TestCase):
    def test_long_text_is_split_without_losing_characters(self):
        text = "абвгд" * 1200
        chunks = bot.split_telegram_text(text, limit=1000)
        self.assertTrue(all(len(chunk) <= 1000 for chunk in chunks))
        self.assertEqual("".join(chunks), text)

    def test_multimodal_text_content_is_extracted(self):
        self.assertEqual(
            bot.extract_text_content([{"text": "Первая часть"}, {"text": "Вторая часть"}]),
            "Первая часть\nВторая часть",
        )

    def test_empty_text_gets_a_retry_message(self):
        self.assertIn("пустой ответ", bot.split_telegram_text("")[0])


if __name__ == "__main__":
    unittest.main()
