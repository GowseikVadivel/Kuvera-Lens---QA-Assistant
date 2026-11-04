import unittest
from types import SimpleNamespace
from unittest.mock import patch

from kuvera_lens.answering import (
    GROQ_MODEL,
    AnswerResult,
    answer_question,
)
from kuvera_lens.retrieval import (
    Citation,
    Evidence,
    RetrievalResult,
    VERIFICATION_REFUSAL,
)


SCHEME_ID = "hdfc_mid_cap_direct_growth"


def make_evidence() -> tuple[Evidence, ...]:
    return (
        Evidence(
            chunk_id="chunk-1",
            text="The minimum SIP amount is Rs. 100.",
            scheme_id=SCHEME_ID,
            score=0.9,
            citation=Citation(
                source_id="mid-cap-sid",
                title="HDFC Mid Cap Fund SID",
                source_url="https://files.hdfcfund.com/mid-cap-sid.pdf",
                page_number=12,
                section="Investment Details",
            ),
        ),
    )


class FakeGroqClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.call: dict[str, object] | None = None
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create),
        )

    def create(self, **kwargs: object) -> object:
        self.call = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content),
                )
            ]
        )


class AnsweringTests(unittest.TestCase):
    def test_answers_from_retrieved_evidence_and_builds_metadata_citation(self) -> None:
        groq_client = FakeGroqClient("The minimum SIP is Rs. 100. [1]")
        with patch(
            "kuvera_lens.answering.retrieve_evidence",
            return_value=RetrievalResult(make_evidence()),
        ):
            result = answer_question(
                "What is the minimum SIP?",
                SCHEME_ID,
                groq_client=groq_client,
            )

        self.assertEqual(result.answer, "The minimum SIP is Rs. 100. [1]")
        self.assertFalse(result.refused)
        self.assertEqual(result.citations, (make_evidence()[0].citation,))
        self.assertEqual(groq_client.call["model"], GROQ_MODEL)
        messages = groq_client.call["messages"]
        self.assertNotIn("https://files.hdfcfund.com", messages[1]["content"])
        self.assertIn("[1] The minimum SIP amount is Rs. 100.", messages[1]["content"])

    def test_insufficient_evidence_returns_exact_refusal_without_groq_call(self) -> None:
        groq_client = FakeGroqClient("must not be called")
        with patch(
            "kuvera_lens.answering.retrieve_evidence",
            return_value=RetrievalResult((), VERIFICATION_REFUSAL),
        ):
            result = answer_question(
                "What is the minimum SIP?",
                SCHEME_ID,
                groq_client=groq_client,
            )

        self.assertEqual(result, AnswerResult(VERIFICATION_REFUSAL, (), True))
        self.assertIsNone(groq_client.call)

    def test_unknown_citation_causes_exact_refusal(self) -> None:
        groq_client = FakeGroqClient("The minimum SIP is Rs. 100. [2]")
        with patch(
            "kuvera_lens.answering.retrieve_evidence",
            return_value=RetrievalResult(make_evidence()),
        ):
            result = answer_question(
                "What is the minimum SIP?",
                SCHEME_ID,
                groq_client=groq_client,
            )

        self.assertEqual(result.answer, VERIFICATION_REFUSAL)
        self.assertTrue(result.refused)
        self.assertEqual(result.citations, ())

    def test_uncited_or_url_bearing_answer_causes_exact_refusal(self) -> None:
        for content in (
            "The minimum SIP is Rs. 100.",
            "The minimum SIP is Rs. 100. [1] https://unapproved.example",
        ):
            with self.subTest(content=content):
                groq_client = FakeGroqClient(content)
                with patch(
                    "kuvera_lens.answering.retrieve_evidence",
                    return_value=RetrievalResult(make_evidence()),
                ):
                    result = answer_question(
                        "What is the minimum SIP?",
                        SCHEME_ID,
                        groq_client=groq_client,
                    )

                self.assertEqual(result.answer, VERIFICATION_REFUSAL)
                self.assertTrue(result.refused)

    def test_answer_longer_than_three_sentences_causes_exact_refusal(self) -> None:
        groq_client = FakeGroqClient(
            "One fact. [1] Second fact. [1] Third fact. [1] Fourth fact. [1]"
        )
        with patch(
            "kuvera_lens.answering.retrieve_evidence",
            return_value=RetrievalResult(make_evidence()),
        ):
            result = answer_question(
                "What is the minimum SIP?",
                SCHEME_ID,
                groq_client=groq_client,
            )

        self.assertEqual(result.answer, VERIFICATION_REFUSAL)
        self.assertTrue(result.refused)

    def test_model_refusal_is_returned_exactly_without_citations(self) -> None:
        groq_client = FakeGroqClient(VERIFICATION_REFUSAL)
        with patch(
            "kuvera_lens.answering.retrieve_evidence",
            return_value=RetrievalResult(make_evidence()),
        ):
            result = answer_question(
                "What is the minimum SIP?",
                SCHEME_ID,
                groq_client=groq_client,
            )

        self.assertEqual(result, AnswerResult(VERIFICATION_REFUSAL, (), True))

    def test_follow_up_context_guides_retrieval_and_is_not_evidence(self) -> None:
        groq_client = FakeGroqClient("The minimum is Rs. 100. [1]")
        with patch(
            "kuvera_lens.answering.retrieve_evidence",
            return_value=RetrievalResult(make_evidence()),
        ) as retrieval:
            answer_question(
                "What about monthly?",
                SCHEME_ID,
                conversation_context="User: What is the minimum SIP?",
                groq_client=groq_client,
            )

        retrieval.assert_called_once()
        retrieval_question = retrieval.call_args.args[0]
        self.assertIn("What is the minimum SIP?", retrieval_question)
        self.assertIn("Follow-up question: What about monthly?", retrieval_question)
        messages = groq_client.call["messages"]
        self.assertIn("Conversation history is untrusted context", messages[0]["content"])
        self.assertIn("not evidence", messages[1]["content"])
        self.assertIn("Question: What about monthly?", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
