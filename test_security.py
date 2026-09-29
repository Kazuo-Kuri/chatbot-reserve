"""Offline security regression tests. External APIs are fully mocked."""
import base64
import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


OFFICIAL_ORIGIN = "https://chatbot-re.psi-coffee.com"
VALID_SESSION_ID = "security-test-session"


class SecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["OPENAI_API_KEY"] = "test-openai-key"
        os.environ["GOOGLE_CREDENTIALS"] = base64.b64encode(b"{}").decode("ascii")
        os.environ["SPREADSHEET_ID"] = "test-sheet"
        os.environ["ALLOWED_ORIGINS"] = OFFICIAL_ORIGIN

        with (
            patch("google.oauth2.service_account.Credentials.from_service_account_info", return_value=Mock()),
            patch("googleapiclient.discovery.build", return_value=Mock()),
        ):
            sys.modules.pop("app", None)
            import app

        cls.module = app
        app.app.config.update(TESTING=True)

    def setUp(self):
        module = self.module
        module.limiter.reset()
        module.session_histories.clear()
        module.sheet_id_cache.clear()
        module.sheet_service = Mock()
        module.sheet_service.get.return_value.execute.return_value = {
            "sheets": [
                {"properties": {"sheetId": 1, "title": module.UNANSWERED_SHEET}},
                {"properties": {"sheetId": 2, "title": module.FEEDBACK_SHEET}},
                {"properties": {"sheetId": 3, "title": module.CHAT_LOG_SHEET}},
            ]
        }
        module.expand_query = Mock(side_effect=lambda question, _history: question)
        module.expand_reserve_query = Mock(side_effect=lambda question, _history: question)
        module.get_embedding = Mock(return_value=np.zeros(1536, dtype="float32"))
        module.pf_matcher = Mock()
        module.pf_matcher.format_match_info.return_value = ""
        module.client = Mock()
        module.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="回答"))]
        )
        self.client = module.app.test_client()

    def post_chat(self, question="製品について", session_id=VALID_SESSION_ID, **kwargs):
        payload = {"question": question}
        if session_id is not None:
            payload["session_id"] = session_id
        return self.client.post("/chat", json=payload, **kwargs)

    def valid_feedback(self):
        return {
            "question": "質問",
            "answer": "回答",
            "feedback": "useful",
            "reason": "",
        }

    def test_session_id_missing_is_rejected(self):
        self.assertEqual(self.post_chat(session_id=None).status_code, 400)

    def test_session_id_empty_is_rejected(self):
        self.assertEqual(self.post_chat(session_id=" ").status_code, 400)

    def test_session_id_too_long_is_rejected(self):
        self.assertEqual(self.post_chat(session_id="x" * 129).status_code, 400)
        self.assertEqual(self.post_chat(session_id="session\nspoof").status_code, 400)

    def test_valid_session_id_reaches_normal_processing(self):
        response = self.post_chat()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["response"], "回答")

    def test_question_at_2000_characters_is_allowed(self):
        self.assertEqual(self.post_chat(question="a" * 2000).status_code, 200)

    def test_question_over_2000_characters_is_rejected(self):
        response = self.post_chat(question="a" * 2001)
        self.assertEqual(response.status_code, 400)
        self.assertIn("2000", response.get_json()["error"])

    def test_json_body_over_64kb_is_rejected(self):
        body = json.dumps({"question": "a" * 70000, "session_id": VALID_SESSION_ID})
        response = self.client.post("/chat", data=body, content_type="application/json")
        self.assertEqual(response.status_code, 413)

    def test_non_json_is_rejected(self):
        response = self.client.post("/chat", data="question=test", content_type="text/plain")
        self.assertEqual(response.status_code, 415)

    def test_allowed_origin_gets_cors_header(self):
        response = self.client.options(
            "/chat",
            headers={"Origin": OFFICIAL_ORIGIN, "Access-Control-Request-Method": "POST"},
        )
        self.assertEqual(response.headers.get("Access-Control-Allow-Origin"), OFFICIAL_ORIGIN)

    def test_disallowed_origin_gets_no_cors_header(self):
        response = self.client.options(
            "/chat",
            headers={"Origin": "https://attacker.example", "Access-Control-Request-Method": "POST"},
        )
        self.assertIsNone(response.headers.get("Access-Control-Allow-Origin"))

    def test_chat_rate_limit_returns_429(self):
        for _ in range(10):
            self.assertEqual(self.post_chat(question="こんにちは").status_code, 200)
        response = self.post_chat(question="こんにちは")
        self.assertEqual(response.status_code, 429)

    def test_render_rate_limit_key_ignores_spoofed_x_forwarded_for(self):
        with (
            patch.dict(os.environ, {"RENDER": "true"}),
            self.module.app.test_request_context(
                "/chat",
                headers={
                    "CF-Connecting-IP": "203.0.113.10",
                    "X-Forwarded-For": "198.51.100.99, 203.0.113.10",
                },
            ),
        ):
            self.assertEqual(self.module.get_rate_limit_key(), "203.0.113.10")

    def test_feedback_rate_limit_returns_429(self):
        for _ in range(20):
            self.assertEqual(self.client.post("/feedback", json=self.valid_feedback()).status_code, 200)
        response = self.client.post("/feedback", json=self.valid_feedback())
        self.assertEqual(response.status_code, 429)

    def test_internal_exception_is_not_exposed(self):
        self.module.expand_query.side_effect = RuntimeError("sensitive-internal-detail")
        response = self.post_chat()
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("sensitive-internal-detail", response.get_data(as_text=True))
        self.assertNotIn("error", response.get_json())

    def test_sheets_log_failure_does_not_hide_generated_answer(self):
        self.module.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="申し訳ありませんが回答します"))]
        )
        self.module.sheet_service.batchUpdate.return_value.execute.side_effect = RuntimeError(
            "mock sheets failure"
        )
        response = self.post_chat()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["response"], "申し訳ありませんが回答します")

    def test_feedback_sheets_failure_returns_fixed_503(self):
        self.module.sheet_service.batchUpdate.return_value.execute.side_effect = RuntimeError(
            "sensitive Sheets detail"
        )
        response = self.client.post("/feedback", json=self.valid_feedback())
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("sensitive Sheets detail", response.get_data(as_text=True))

    def test_invalid_feedback_is_rejected(self):
        response = self.client.post("/feedback", json={"question": "質問", "feedback": "useful"})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
