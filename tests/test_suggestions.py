import unittest
from uuid import uuid4

import numpy as np
from qdrant_client import QdrantClient, models

from kuvera_lens.answering import answer_question
from kuvera_lens.qdrant_store import ACTIVE_COLLECTION_ALIAS
from kuvera_lens.retrieval import retrieve_evidence
from kuvera_lens.schemes import SUPPORTED_SCHEMES
from kuvera_lens.suggestions import SuggestedQuestion, get_suggested_questions

COLLECTION_NAME = "suggestions_test_collection"
_SCHEME_EVIDENCE = (
    (
        "Minimum SIP is Rs. 100. Exit load is 1% within one year.",
        "Riskometer is Very High. The investment objective is long-term capital appreciation.",
    ),
    (
        "Minimum SIP is Rs. 500.",
        "The scheme lock-in period is not applicable.",
    ),
    (
        "Exit load is 1% within one year.",
        "Riskometer is Very High.",
    ),
    (
        "Riskometer is Very High.",
        "The investment objective is long-term capital appreciation.",
    ),
)


class FakeGroqClient:
    def __init__(self) -> None:
        self.chat = self
        self.completions = self

    def create(self, **kwargs: object) -> object:
        return type(
            "Response",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "message": type(
                                "Message",
                                (),
                                {"content": "The official evidence provides this detail [1]."},
                            )()
                        },
                    )()
                ]
            },
        )()


class FakeEmbeddingModel:
    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        return np.array([[1.0, 0.0]], dtype=np.float32)


class SuggestedQuestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=2,
                distance=models.Distance.COSINE,
            ),
        )
        points = []
        for scheme, (webpage_text, sid_text) in zip(
            SUPPORTED_SCHEMES, _SCHEME_EVIDENCE
        ):
            for suffix, document_type, domain, url, title, page, text in (
                (
                    "webpage",
                    "scheme_page",
                    "www.hdfcfund.com",
                    f"https://www.hdfcfund.com/{scheme.scheme_id}",
                    f"{scheme.display_name} scheme page",
                    None,
                    webpage_text,
                ),
                (
                    "sid",
                    "SID",
                    "files.hdfcfund.com",
                    f"https://files.hdfcfund.com/{scheme.scheme_id}.pdf",
                    f"{scheme.display_name} SID",
                    4,
                    sid_text,
                ),
            ):
                digest = (scheme.scheme_id.encode().hex() + "a" * 64)[:64]
                points.append(
                    models.PointStruct(
                        id=uuid4(),
                        vector=[1.0, 0.0],
                        payload={
                            "chunk_id": f"{scheme.scheme_id}-{suffix}",
                            "source_id": f"{scheme.scheme_id}-{suffix}",
                            "scheme_id": scheme.scheme_id,
                            "scheme_name": scheme.display_name,
                            "source_url": url,
                            "source_domain": domain,
                            "document_title": title,
                            "document_type": document_type,
                            "document_version": digest,
                            "content_hash": digest,
                            "page_number": page,
                            "section": "Scheme details",
                            "text": text,
                        },
                    )
                )
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        self.client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=COLLECTION_NAME,
                        alias_name=ACTIVE_COLLECTION_ALIAS,
                    )
                )
            ]
        )
    def tearDown(self) -> None:
        self.client.close()

    def test_suggestions_for_all_schemes_use_evidence_and_answer_with_citations(self) -> None:
        results: dict[str, tuple[SuggestedQuestion, ...]] = {}
        for scheme in SUPPORTED_SCHEMES:
            with self.subTest(scheme=scheme.scheme_id):
                questions = get_suggested_questions(
                    scheme.scheme_id,
                    client=self.client,
                )
                self.assertGreater(len(questions), 0)
                self.assertLessEqual(len(questions), 3)

                supporting_source_types: set[str] = set()
                for suggestion in questions:
                    evidence = retrieve_evidence(
                        suggestion.question,
                        scheme.scheme_id,
                        client=self.client,
                        source_type=suggestion.source_type,
                        model=FakeEmbeddingModel(),
                    )
                    self.assertTrue(evidence.evidence)
                    supporting_source_types.add(suggestion.source_type)

                    answer = answer_question(
                        suggestion.question,
                        scheme.scheme_id,
                        source_type=suggestion.source_type,
                        qdrant_client=self.client,
                        embedding_model=FakeEmbeddingModel(),
                        groq_client=FakeGroqClient(),
                    )
                    self.assertFalse(answer.refused)
                    self.assertTrue(answer.citations)
                    self.assertTrue(all(
                        suggestion.source_type == "webpage"
                        and "www.hdfcfund.com" in item.source_url
                        or suggestion.source_type == "sid_pdf"
                        and "files.hdfcfund.com" in item.source_url
                        for item in answer.citations
                    ))

                self.assertTrue(supporting_source_types)
                results[scheme.scheme_id] = questions

    def test_scheme_specific_suggestions_do_not_leak_between_schemes(self) -> None:
        large_cap = get_suggested_questions(
            "hdfc_large_cap_direct_growth",
            client=self.client,
        )
        flexi_cap = get_suggested_questions(
            "hdfc_flexi_cap_direct_growth",
            client=self.client,
        )
        large_questions = {suggestion.question for suggestion in large_cap}
        flexi_questions = {suggestion.question for suggestion in flexi_cap}
        self.assertTrue(all("Large Cap" in question for question in large_questions))
        self.assertTrue(all("Flexi Cap" in question for question in flexi_questions))
        self.assertFalse(large_questions & flexi_questions)

    def test_suggestions_use_one_payload_scan_without_embedding_queries(self) -> None:
        class CountingClient:
            def __init__(self, client: QdrantClient) -> None:
                self.client = client
                self.scroll_calls = 0
                self.query_calls = 0

            def scroll(self, *args: object, **kwargs: object) -> tuple[list[object], object | None]:
                self.scroll_calls += 1
                return self.client.scroll(*args, **kwargs)

            def query_points(self, *args: object, **kwargs: object) -> object:
                self.query_calls += 1
                return self.client.query_points(*args, **kwargs)

        client = CountingClient(self.client)
        suggestions = get_suggested_questions(
            "hdfc_large_cap_direct_growth",
            client=client,  # type: ignore[arg-type]
        )
        self.assertTrue(suggestions)
        self.assertEqual(client.scroll_calls, 1)
        self.assertEqual(client.query_calls, 0)


if __name__ == "__main__":
    unittest.main()
