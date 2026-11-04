import tempfile
import unittest
from uuid import uuid4
from unittest.mock import patch

import numpy as np
from qdrant_client import QdrantClient, models

from kuvera_lens.qdrant_store import ACTIVE_COLLECTION_ALIAS
from kuvera_lens.retrieval import (
    DEFAULT_MINIMUM_COSINE_SIMILARITY,
    UnsupportedSchemeError,
    VERIFICATION_REFUSAL,
    _load_retrieval_model,
    retrieve_evidence,
)

SCHEME_ID = "hdfc_mid_cap_direct_growth"
OTHER_SCHEME_ID = "hdfc_large_cap_direct_growth"
COLLECTION_NAME = "retrieval_test_collection"


class FakeEmbeddingModel:
    def __init__(self, vector: tuple[float, ...] = (1.0, 0.0)) -> None:
        self.vector = np.asarray([vector], dtype=np.float32)
        self.calls: list[list[str]] = []

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        self.asserted_options = (
            batch_size,
            show_progress_bar,
            convert_to_numpy,
            normalize_embeddings,
        )
        self.calls.append(sentences)
        return self.vector.copy()


def make_payload(
    *,
    chunk_id: str = "chunk-1",
    scheme_id: str = SCHEME_ID,
    source_id: str = "mid-cap-sid",
    source_url: str = "https://files.hdfcfund.com/mid-cap-sid.pdf",
    source_domain: str = "files.hdfcfund.com",
    document_title: str = "HDFC Mid Cap Fund SID",
    document_type: str = "SID",
    document_version: str = "a" * 64,
    content_hash: str = "a" * 64,
    page_number: int | None = 12,
    text: str = "The minimum SIP amount is Rs. 100.",
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "source_id": source_id,
        "scheme_id": scheme_id,
        "scheme_name": "HDFC Mid Cap Fund – Direct Growth",
        "source_url": source_url,
        "source_domain": source_domain,
        "document_title": document_title,
        "document_type": document_type,
        "document_version": document_version,
        "content_hash": content_hash,
        "page_number": page_number,
        "section": "Investment Details",
        "text": text,
    }


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=2,
                distance=models.Distance.COSINE,
            ),
        )
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=uuid4(),
                    vector=[1.0, 0.0],
                    payload=make_payload(chunk_id="relevant"),
                ),
                models.PointStruct(
                    id=uuid4(),
                    vector=[1.0, 0.0],
                    payload=make_payload(
                        chunk_id="mid-cap-webpage",
                        source_id="mid-cap-direct",
                        source_url="https://www.hdfcfund.com/explore/mutual-funds/hdfc-mid-cap-fund/direct",
                        source_domain="www.hdfcfund.com",
                        document_title="HDFC Mid Cap Fund",
                        document_type="scheme_page",
                        page_number=None,
                    ),
                ),
                models.PointStruct(
                    id=uuid4(),
                    vector=[1.0, 0.0],
                    payload=make_payload(
                        chunk_id="other-scheme",
                        scheme_id=OTHER_SCHEME_ID,
                        source_id="large-cap-sid",
                        source_url="https://files.hdfcfund.com/large-cap-sid.pdf",
                        document_title="HDFC Large Cap Fund SID",
                    ),
                ),
                models.PointStruct(
                    id=uuid4(),
                    vector=[1.0, 0.0],
                    payload=make_payload(
                        chunk_id="large-cap-webpage",
                        scheme_id=OTHER_SCHEME_ID,
                        source_id="large-cap-direct",
                        source_url="https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct",
                        source_domain="www.hdfcfund.com",
                        document_title="HDFC Large Cap Fund",
                        document_type="scheme_page",
                        page_number=None,
                    ),
                ),
                models.PointStruct(
                    id=uuid4(),
                    vector=[1.0, 0.0],
                    payload=make_payload(
                        chunk_id="unapproved",
                        source_url="https://attacker.example/fund.pdf",
                        source_domain="attacker.example",
                    ),
                ),
                models.PointStruct(
                    id=uuid4(),
                    vector=[0.0, 1.0],
                    payload=make_payload(chunk_id="irrelevant"),
                ),
            ],
        )
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
        self.temporary_directory.cleanup()

    def test_returns_relevant_approved_evidence_for_selected_scheme(self) -> None:
        model = FakeEmbeddingModel()

        result = retrieve_evidence(
            "What is the minimum SIP?",
            SCHEME_ID,
            client=self.client,
            model=model,
            source_type="sid_pdf",
        )

        self.assertFalse(result.is_refusal)
        self.assertEqual(len(result.evidence), 1)
        evidence = result.evidence[0]
        self.assertEqual(evidence.chunk_id, "relevant")
        self.assertEqual(evidence.scheme_id, SCHEME_ID)
        self.assertGreaterEqual(
            evidence.score,
            DEFAULT_MINIMUM_COSINE_SIMILARITY,
        )
        self.assertEqual(evidence.citation.title, "HDFC Mid Cap Fund SID")
        self.assertEqual(evidence.citation.page_number, 12)
        self.assertEqual(
            evidence.citation.source_url,
            "https://files.hdfcfund.com/mid-cap-sid.pdf",
        )
        self.assertEqual(model.calls, [["What is the minimum SIP?"]])

    def test_webpage_only_retrieval_returns_webpage_evidence(self) -> None:
        result = retrieve_evidence(
            "What is the minimum SIP?",
            SCHEME_ID,
            client=self.client,
            model=FakeEmbeddingModel(),
            source_type="webpage",
        )

        self.assertEqual([item.chunk_id for item in result.evidence], ["mid-cap-webpage"])
        self.assertIsNone(result.evidence[0].citation.page_number)
        self.assertEqual(
            result.evidence[0].citation.source_url,
            "https://www.hdfcfund.com/explore/mutual-funds/hdfc-mid-cap-fund/direct",
        )

    def test_sid_only_retrieval_returns_sid_evidence_and_page_citation(self) -> None:
        result = retrieve_evidence(
            "What is the minimum SIP?",
            SCHEME_ID,
            client=self.client,
            model=FakeEmbeddingModel(),
            source_type="sid_pdf",
        )

        self.assertEqual([item.chunk_id for item in result.evidence], ["relevant"])
        self.assertEqual(result.evidence[0].citation.title, "HDFC Mid Cap Fund SID")
        self.assertEqual(result.evidence[0].citation.page_number, 12)

    def test_sid_filter_excludes_kim_evidence(self) -> None:
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=uuid4(),
                    vector=[1.0, 0.0],
                    payload=make_payload(
                        chunk_id="kim-summary",
                        source_id="mid-cap-kim",
                        document_title="KIM - HDFC Mid Cap Fund",
                        document_type="SID",
                    ),
                )
            ],
        )

        result = retrieve_evidence(
            "What is the minimum SIP?",
            SCHEME_ID,
            client=self.client,
            model=FakeEmbeddingModel(),
            source_type="sid_pdf",
        )

        self.assertEqual([item.chunk_id for item in result.evidence], ["relevant"])

    def test_exact_webpage_ter_field_can_pass_the_semantic_cutoff(self) -> None:
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=uuid4(),
                    vector=[0.0, 1.0],
                    payload=make_payload(
                        chunk_id="ter-field",
                        source_id="mid-cap-direct",
                        source_url="https://www.hdfcfund.com/explore/mutual-funds/hdfc-mid-cap-fund/direct",
                        source_domain="www.hdfcfund.com",
                        document_title="HDFC Mid Cap Fund",
                        document_type="scheme_page",
                        page_number=None,
                        text="Current TER 0.74 Lock in NA.",
                    ),
                )
            ],
        )

        result = retrieve_evidence(
            "What is the current expense ratio?",
            SCHEME_ID,
            client=self.client,
            model=FakeEmbeddingModel(),
            source_type="webpage",
        )

        self.assertIn("ter-field", [item.chunk_id for item in result.evidence])
        field_evidence = next(item for item in result.evidence if item.chunk_id == "ter-field")
        self.assertLess(field_evidence.score, DEFAULT_MINIMUM_COSINE_SIMILARITY)

    def test_exact_webpage_riskometer_field_can_pass_the_semantic_cutoff(self) -> None:
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=uuid4(),
                    vector=[0.0, 1.0],
                    payload=make_payload(
                        chunk_id="riskometer-field",
                        source_id="mid-cap-direct",
                        source_url="https://www.hdfcfund.com/explore/mutual-funds/hdfc-mid-cap-fund/direct",
                        source_domain="www.hdfcfund.com",
                        document_title="HDFC Mid Cap Fund",
                        document_type="scheme_page",
                        page_number=None,
                        text="Riskometer Very High Min SIP Rs. 100.",
                    ),
                )
            ],
        )

        result = retrieve_evidence(
            "What is the riskometer level?",
            SCHEME_ID,
            client=self.client,
            model=FakeEmbeddingModel(),
            source_type="webpage",
        )

        self.assertIn("riskometer-field", [item.chunk_id for item in result.evidence])
        field_evidence = next(
            item for item in result.evidence if item.chunk_id == "riskometer-field"
        )
        self.assertLess(field_evidence.score, DEFAULT_MINIMUM_COSINE_SIMILARITY)

    def test_exact_sid_allocation_table_is_prioritized_below_similarity_cutoff(self) -> None:
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=uuid4(),
                    vector=[0.0, 1.0],
                    payload=make_payload(
                        chunk_id="large-cap-allocation",
                        scheme_id=OTHER_SCHEME_ID,
                        source_id="large-cap-sid",
                        source_url="https://files.hdfcfund.com/large-cap-sid.pdf",
                        document_title="HDFC Large Cap Fund SID",
                        page_number=11,
                        text=(
                            "A. HOW WILL THE SCHEME ALLOCATE ITS ASSETS? "
                            "Instruments Indicative allocations (% of total assets) "
                            "Minimum Maximum Equity and Equity Related Instruments "
                            "of Large Cap Companies 80 100. Debt Securities "
                            "0 20."
                        ),
                    ),
                )
            ],
        )

        result = retrieve_evidence(
            "What are the asset allocation limits?",
            OTHER_SCHEME_ID,
            client=self.client,
            model=FakeEmbeddingModel(),
            source_type="sid_pdf",
            top_k=1,
        )

        self.assertEqual(
            [item.chunk_id for item in result.evidence],
            ["large-cap-allocation"],
        )
        evidence = result.evidence[0]
        self.assertLess(evidence.score, DEFAULT_MINIMUM_COSINE_SIMILARITY)
        self.assertEqual(evidence.citation.page_number, 11)
        self.assertEqual(
            evidence.citation.source_url,
            "https://files.hdfcfund.com/large-cap-sid.pdf",
        )

    def test_fact_questions_use_source_vocabulary_for_embedding(self) -> None:
        cases = (
            ("What is the current expense ratio?", "What TER does this mutual fund charge?"),
            ("What is the riskometer level?", "What is the scheme Riskometer level?"),
            (
                "What are the asset allocation limits?",
                "What are the indicative asset allocations and minimum and maximum percentages by instrument?",
            ),
            (
                "What are the principal risks associated with this fund?",
                "What are the principal risks and risk factors associated with the scheme's investments?",
            ),
        )
        for question, expected_query in cases:
            with self.subTest(question=question):
                model = FakeEmbeddingModel()
                retrieve_evidence(
                    question,
                    SCHEME_ID,
                    client=self.client,
                    model=model,
                )
                self.assertEqual(model.calls, [[expected_query]])

    def test_combined_retrieval_can_return_webpage_and_sid_evidence(self) -> None:
        result = retrieve_evidence(
            "What is the minimum SIP?",
            SCHEME_ID,
            client=self.client,
            model=FakeEmbeddingModel(),
            source_type="both",
            top_k=10,
        )

        self.assertEqual(
            {item.chunk_id for item in result.evidence},
            {"relevant", "mid-cap-webpage"},
        )
        self.assertEqual(
            {item.citation.title for item in result.evidence},
            {"HDFC Mid Cap Fund SID", "HDFC Mid Cap Fund"},
        )

    def test_source_type_retrieval_remains_isolated_by_scheme(self) -> None:
        for scheme_id, expected_chunks in (
            (SCHEME_ID, {"relevant", "mid-cap-webpage"}),
            (OTHER_SCHEME_ID, {"other-scheme", "large-cap-webpage"}),
        ):
            with self.subTest(scheme_id=scheme_id):
                result = retrieve_evidence(
                    "What is the minimum SIP?",
                    scheme_id,
                    client=self.client,
                    model=FakeEmbeddingModel(),
                    source_type="both",
                    top_k=10,
                )

                self.assertEqual(
                    {item.chunk_id for item in result.evidence},
                    expected_chunks,
                )
                self.assertTrue(
                    all(item.scheme_id == scheme_id for item in result.evidence)
                )

    def test_returns_exact_refusal_when_no_result_meets_threshold(self) -> None:
        model = FakeEmbeddingModel((-1.0, 0.0))

        result = retrieve_evidence(
            "What is the minimum SIP?",
            SCHEME_ID,
            client=self.client,
            model=model,
        )

        self.assertTrue(result.is_refusal)
        self.assertEqual(result.refusal, VERIFICATION_REFUSAL)
        self.assertEqual(result.evidence, ())

    def test_rejects_unsupported_scheme_before_embedding_or_query(self) -> None:
        model = FakeEmbeddingModel()

        with self.assertRaises(UnsupportedSchemeError):
            retrieve_evidence(
                "What is the minimum SIP?",
                "unsupported-scheme",
                client=self.client,
                model=model,
            )

        self.assertEqual(model.calls, [])

    def test_rejects_empty_question_before_embedding(self) -> None:
        model = FakeEmbeddingModel()

        with self.assertRaisesRegex(ValueError, "non-empty"):
            retrieve_evidence("  ", SCHEME_ID, client=self.client, model=model)

        self.assertEqual(model.calls, [])

    def test_rejects_invalid_embedding_vector(self) -> None:
        model = FakeEmbeddingModel((0.0, 0.0))

        with self.assertRaisesRegex(RuntimeError, "non-normalized"):
            retrieve_evidence(
                "What is the minimum SIP?",
                SCHEME_ID,
                client=self.client,
                model=model,
            )

    def test_default_embedding_model_is_loaded_once_and_reused(self) -> None:
        _load_retrieval_model.cache_clear()
        try:
            with patch(
                "kuvera_lens.retrieval._load_model",
                return_value=FakeEmbeddingModel(),
            ) as load_model:
                first_model = _load_retrieval_model()
                second_model = _load_retrieval_model()

            self.assertIs(first_model, second_model)
            load_model.assert_called_once_with()
        finally:
            _load_retrieval_model.cache_clear()


if __name__ == "__main__":
    unittest.main()
