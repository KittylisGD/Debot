import unittest

from bot import MODES, MAX_HISTORY_MESSAGES, build_ollama_messages, split_telegram_text


class OllamaMessageTests(unittest.TestCase):
    def test_selected_persona_is_sent_as_system_message(self):
        messages = build_ollama_messages("Мемный 😂", [], "Привет")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], MODES["Мемный 😂"])
        self.assertEqual(messages[-1], {"role": "user", "content": "Привет"})

    def test_unknown_mode_uses_default_persona(self):
        messages = build_ollama_messages("неизвестно", [], "Тест")
        self.assertEqual(messages[0]["content"], MODES["Дебил 🤪"])

    def test_image_is_attached_to_local_user_message(self):
        messages = build_ollama_messages("Злой 😈", [], "Что на фото?", "aGVsbG8=")
        self.assertEqual(messages[-1]["images"], ["aGVsbG8="])
        self.assertEqual(messages[-1]["content"], "Что на фото?")

    def test_history_is_limited_to_recent_messages(self):
        history = [{"role": "user", "content": str(i)} for i in range(MAX_HISTORY_MESSAGES + 4)]
        messages = build_ollama_messages("Дебил 🤪", history, "новое")
        self.assertEqual(len(messages), MAX_HISTORY_MESSAGES + 2)  # system + history + current
        self.assertEqual(messages[1]["content"], "4")


class TelegramTextTests(unittest.TestCase):
    def test_long_text_is_split_without_losing_characters(self):
        text = "абвгд" * 1200
        chunks = split_telegram_text(text, limit=1000)
        self.assertTrue(all(len(chunk) <= 1000 for chunk in chunks))
        self.assertEqual("".join(chunks), text)

    def test_empty_text_gets_a_real_retry_message(self):
        self.assertIn("пустой ответ", split_telegram_text("")[0])


if __name__ == "__main__":
    unittest.main()
