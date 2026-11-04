import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from groq import GroqError, RateLimitError
from streamlit.testing.v1 import AppTest

from kuvera_lens.answering import AnswerResult
from kuvera_lens.retrieval import Citation, VERIFICATION_REFUSAL
from kuvera_lens.schemes import SUPPORTED_SCHEMES
from kuvera_lens.suggestions import SuggestedQuestion

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def make_app() -> AppTest:
    return AppTest.from_file(str(APP_PATH), default_timeout=10).run()


class StreamlitAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.suggestions_patch = patch(
            "kuvera_lens.ui.get_suggested_questions",
            return_value=(),
        )
        self.suggestions_patch.start()
        self.addCleanup(self.suggestions_patch.stop)

    def test_each_supported_scheme_can_start_and_change_conversation(self) -> None:
        for scheme in SUPPORTED_SCHEMES:
            with self.subTest(scheme=scheme.scheme_id):
                app = make_app()
                app.button(key=f"select-{scheme.scheme_id}").click().run()
                self.assertEqual(app.session_state["lens_scheme_id"], scheme.scheme_id)
                self.assertEqual(app.session_state["lens_screen"], "selection")

                app.button(key="start-chat").click().run()
                self.assertEqual(app.session_state["lens_screen"], "conversation")
                self.assertEqual(app.session_state["lens_scheme_id"], scheme.scheme_id)
                self.assertEqual(len(app.chat_input), 1)
                self.assertFalse(app.exception)

                app.button(key="change-scheme").click().run()
                self.assertEqual(app.session_state["lens_screen"], "selection")
                self.assertIsNone(app.session_state["lens_scheme_id"])
                self.assertEqual(app.session_state["lens_messages"], [])

    def test_question_uses_backend_and_displays_only_returned_citation(self) -> None:
        citation = Citation(
            source_id="mid-cap-sid",
            title=(
                "Invest in the HDFC Mid Cap Fund Direct Plan Mutual Fund | "
                "HDFC Mutual Fund | page content that should not be shown"
            ),
            source_url="https://files.hdfcfund.com/mid-cap-sid.pdf",
            page_number=12,
            section="Long scraped page text that should not appear in the citation button",
        )
        answer = AnswerResult("The minimum SIP is Rs. 100. [1]", (citation,), False)
        scheme = SUPPORTED_SCHEMES[0]
        with patch("kuvera_lens.ui.answer_question", return_value=answer) as answer_service:
            app = make_app()
            app.button(key=f"select-{scheme.scheme_id}").click().run()
            app.button(key="start-chat").click().run()
            app.chat_input[0].set_value("What is the minimum SIP?").run()

        answer_service.assert_called_once_with(
            "What is the minimum SIP?",
            scheme.scheme_id,
            conversation_context=None,
        )
        self.assertEqual(len(app.session_state["lens_messages"]), 2)
        self.assertEqual(app.session_state["lens_messages"][1]["content"], answer.answer)
        self.assertEqual(
            app.session_state["lens_messages"][1]["citations"][0]["source_url"],
            citation.source_url,
        )
        self.assertEqual(len(app.link_button), 1)
        self.assertIn("p. 12", app.link_button[0].label)
        self.assertLess(len(app.link_button[0].label), 60)
        self.assertNotIn("scraped", app.link_button[0].label)
        self.assertEqual(len(app.exception), 0)

    def test_insufficient_evidence_refusal_is_rendered_without_a_citation(self) -> None:
        refusal = AnswerResult(VERIFICATION_REFUSAL, (), True)
        scheme = SUPPORTED_SCHEMES[1]
        with patch("kuvera_lens.ui.answer_question", return_value=refusal):
            app = make_app()
            app.button(key=f"select-{scheme.scheme_id}").click().run()
            app.button(key="start-chat").click().run()
            app.chat_input[0].set_value("Question not covered by the documents").run()

        self.assertEqual(
            app.session_state["lens_messages"][1]["content"],
            VERIFICATION_REFUSAL,
        )
        self.assertTrue(app.session_state["lens_messages"][1]["refused"])
        self.assertEqual(len(app.link_button), 0)
        self.assertEqual(len(app.exception), 0)

    def test_groq_authentication_failure_has_actionable_safe_message(self) -> None:
        class AuthenticationError(GroqError):
            pass

        scheme = SUPPORTED_SCHEMES[2]
        with patch(
            "kuvera_lens.ui.answer_question",
            side_effect=AuthenticationError("provider detail must not be shown"),
        ):
            app = make_app()
            app.button(key=f"select-{scheme.scheme_id}").click().run()
            app.button(key="start-chat").click().run()
            app.chat_input[0].set_value("What is the lock-in period?").run()

        error_message = app.session_state["lens_messages"][-1]["content"]
        self.assertIn("GROQ_API_KEY", error_message)
        self.assertNotIn("provider detail", error_message)
        self.assertEqual(len(app.exception), 0)

    def test_groq_rate_limit_shows_retry_after_without_provider_details(self) -> None:
        rate_limit = RateLimitError(
            "provider detail must not be shown",
            response=httpx.Response(
                429,
                request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
                headers={"retry-after": "287"},
            ),
            body={"error": "provider detail must not be shown"},
        )
        scheme = SUPPORTED_SCHEMES[2]
        with patch("kuvera_lens.ui.answer_question", side_effect=rate_limit):
            app = make_app()
            app.button(key=f"select-{scheme.scheme_id}").click().run()
            app.button(key="start-chat").click().run()
            app.chat_input[0].set_value("What is the lock-in period?").run()

        error_message = app.session_state["lens_messages"][-1]["content"]
        self.assertEqual(
            error_message,
            "The Groq service is rate-limiting requests. Please try again in about 5 minutes.",
        )
        self.assertNotIn("provider detail", error_message)
        self.assertEqual(len(app.exception), 0)

    def test_follow_up_question_receives_bounded_conversation_context(self) -> None:
        citation = Citation(
            source_id="large-cap-sid",
            title="HDFC Large Cap Fund SID",
            source_url="https://files.hdfcfund.com/large-cap-sid.pdf",
            page_number=8,
            section="Investment Details",
        )
        first = AnswerResult("The minimum SIP is Rs. 100. [1]", (citation,), False)
        second = AnswerResult("The additional SIP is Rs. 100. [1]", (citation,), False)
        scheme = SUPPORTED_SCHEMES[0]
        with patch(
            "kuvera_lens.ui.answer_question",
            side_effect=(first, second),
        ) as answer_service:
            app = make_app()
            app.button(key=f"select-{scheme.scheme_id}").click().run()
            app.button(key="start-chat").click().run()
            app.chat_input[0].set_value("What is the minimum SIP?").run()
            app.chat_input[0].set_value("What about monthly?").run()

        self.assertEqual(answer_service.call_count, 2)
        context = answer_service.call_args.kwargs["conversation_context"]
        self.assertIn("What is the minimum SIP?", context)
        self.assertIn("The minimum SIP is Rs. 100. [1]", context)
        self.assertEqual(len(app.session_state["lens_messages"]), 4)
        self.assertEqual(len(app.link_button), 2)
        self.assertEqual(len(app.exception), 0)

    def test_suggested_questions_refresh_when_scheme_changes(self) -> None:
        first_scheme, second_scheme = SUPPORTED_SCHEMES[:2]
        first_suggestion = SuggestedQuestion(
            "What is the Large Cap investment objective?",
            "sid_pdf",
        )
        second_suggestion = SuggestedQuestion(
            "What is the Flexi Cap lock-in period?",
            "sid_pdf",
        )
        answer = AnswerResult("The official evidence supports this answer [1].", (), False)
        with (
            patch("kuvera_lens.ui.answer_question", return_value=answer) as answer_service,
            patch(
                "kuvera_lens.ui.get_suggested_questions",
                side_effect=(
                    (first_suggestion,),
                    (second_suggestion,),
                ),
            ) as suggestion_service,
        ):
            app = make_app()
            app.button(key=f"select-{first_scheme.scheme_id}").click().run()
            app.button(key="start-chat").click().run()
            app.chat_input[0].set_value("Give me a supported scheme fact.").run()

            self.assertEqual(
                app.session_state["lens_suggestions"],
                [first_suggestion],
            )
            app.button(key=f"suggest-2-{first_suggestion.question}").click().run()
            app.button(key="new-conversation").click().run()
            app.button(key=f"select-{second_scheme.scheme_id}").click().run()
            app.button(key="start-chat").click().run()
            app.chat_input[0].set_value("Give me a supported scheme fact.").run()

        self.assertEqual(suggestion_service.call_args_list[0].args, (first_scheme.scheme_id,))
        self.assertEqual(suggestion_service.call_args_list[1].args, (second_scheme.scheme_id,))
        self.assertEqual(
            app.session_state["lens_suggestions"],
            [second_suggestion],
        )
        self.assertNotIn(first_suggestion, app.session_state["lens_suggestions"])
        self.assertEqual(answer_service.call_count, 3)
        self.assertEqual(
            answer_service.call_args_list[1].kwargs["source_type"],
            first_suggestion.source_type,
        )
        self.assertNotIn("source_type", answer_service.call_args_list[2].kwargs)
        self.assertEqual(answer_service.call_args_list[1].args[1], first_scheme.scheme_id)
        self.assertEqual(answer_service.call_args_list[2].args[1], second_scheme.scheme_id)
        self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
