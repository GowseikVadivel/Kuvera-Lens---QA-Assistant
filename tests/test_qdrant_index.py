import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np
from qdrant_client import QdrantClient, models

from kuvera_lens.embedding import MODEL_NAME, MODEL_PROFILE
from kuvera_lens.qdrant_index import (
    COLLECTION_PREFIX,
    index_active_embeddings,
)
from kuvera_lens.qdrant_store import ACTIVE_COLLECTION_ALIAS


SOURCE_ID = "mid-cap-sid"
SCHEME_ID = "hdfc_mid_cap_direct_growth"
CONTENT_HASH = "a" * 64


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_index_inputs(
    root: Path,
    *,
    chunk_ids: tuple[str, ...] = ("chunk-1", "chunk-2"),
    content_hash: str = CONTENT_HASH,
) -> tuple[Path, Path]:
    chunk_root = root / "chunks"
    embedding_root = root / "embeddings"
    chunk_root.mkdir(parents=True, exist_ok=True)
    embedding_root.mkdir(parents=True, exist_ok=True)
    chunk_relative_path = Path(SCHEME_ID) / f"{SOURCE_ID}-{content_hash[:12]}.jsonl"
    chunk_file = chunk_root / chunk_relative_path
    chunk_file.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "chunk_id": chunk_id,
            "text": f"Facts for {chunk_id}.",
            "metadata": {
                "source_id": SOURCE_ID,
                "scheme_id": SCHEME_ID,
                "scheme_name": "HDFC Mid Cap Fund – Direct Growth",
                "source_url": "https://files.hdfcfund.com/mid-cap-sid.pdf",
                "source_domain": "files.hdfcfund.com",
                "document_title": "HDFC Mid Cap Fund SID",
                "document_type": "SID",
                "content_hash": content_hash,
                "document_version": content_hash,
                "last_checked_at": "2026-11-03T00:00:00+00:00",
                "page_number": index + 1,
                "section": "Investment details",
            },
        }
        for index, chunk_id in enumerate(chunk_ids)
    ]
    chunk_file.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    chunk_id_digest = hashlib.sha256()
    for chunk_id in chunk_ids:
        chunk_id_digest.update(chunk_id.encode("utf-8"))
        chunk_id_digest.update(b"\0")

    chunk_manifest = chunk_root / "active_chunks.json"
    chunk_manifest.write_text(
        json.dumps(
            {
                "documents": 1,
                "chunks": len(rows),
                "files": [
                    {
                        "source_id": SOURCE_ID,
                        "scheme_id": SCHEME_ID,
                        "content_hash": content_hash,
                        "chunk_count": len(rows),
                        "path": chunk_relative_path.as_posix(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    vector_relative_path = Path("files") / f"{SOURCE_ID}-{content_hash[:12]}.npz"
    vector_file = embedding_root / vector_relative_path
    vector_file.parent.mkdir(parents=True, exist_ok=True)
    vectors = np.tile(
        np.asarray([[0.6, 0.8]], dtype=np.float32),
        (len(rows), 1),
    )
    with vector_file.open("wb") as file:
        np.savez_compressed(
            file,
            chunk_ids=np.asarray(chunk_ids, dtype=np.str_),
            embeddings=vectors,
        )
    embedding_manifest = embedding_root / "active_embeddings.json"
    embedding_manifest.write_text(
        json.dumps(
            {
                "model_name": MODEL_NAME,
                "model_profile": MODEL_PROFILE,
                "normalized": True,
                "dtype": "float32",
                "dimensions": 2,
                "chunks": len(rows),
                "documents": [
                    {
                        "source_id": SOURCE_ID,
                        "scheme_id": SCHEME_ID,
                        "content_hash": content_hash,
                        "chunk_id_digest": chunk_id_digest.hexdigest(),
                        "chunk_file": chunk_relative_path.as_posix(),
                        "chunk_count": len(rows),
                        "path": vector_relative_path.as_posix(),
                        "file_sha256": _sha256(vector_file.read_bytes()),
                        "model_profile": MODEL_PROFILE,
                        "dimensions": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return chunk_manifest, embedding_manifest


class FailingUpsertClient:
    def __init__(self, client: QdrantClient) -> None:
        self._client = client

    def __getattr__(self, name: str):
        return getattr(self._client, name)

    def upsert(self, **kwargs):
        raise OSError("simulated Qdrant write failure")


class QdrantIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        prefix = f"{COLLECTION_PREFIX}{uuid4().hex[:8]}_"
        self.prefix_patch = patch("kuvera_lens.qdrant_index.COLLECTION_PREFIX", prefix)
        self.prefix_patch.start()
        self.addCleanup(self.prefix_patch.stop)
        self.client = QdrantClient(":memory:")
        self.chunk_manifest, self.embedding_manifest = write_index_inputs(self.root)

    def tearDown(self) -> None:
        self.client.close()
        self.temporary_directory.cleanup()

    def test_indexes_vectors_text_and_citation_metadata(self) -> None:
        with patch.object(
            self.client,
            "create_payload_index",
            return_value=True,
        ) as create_payload_index:
            stats = index_active_embeddings(
                self.client,
                self.chunk_manifest,
                self.embedding_manifest,
                batch_size=1,
            )

        self.assertTrue(stats.updated)
        self.assertEqual(stats.chunks_indexed, 2)
        self.assertEqual(stats.schemes_indexed, 1)
        self.assertEqual(stats.collection_alias, ACTIVE_COLLECTION_ALIAS)
        create_payload_index.assert_called_once_with(
            collection_name=stats.collection_name,
            field_name="scheme_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )
        self.assertEqual(stats.collection_name, self.client.get_aliases().aliases[0].collection_name)
        records, offset = self.client.scroll(
            collection_name=ACTIVE_COLLECTION_ALIAS,
            limit=10,
            with_payload=True,
            with_vectors=True,
        )
        self.assertIsNone(offset)
        self.assertEqual(len(records), 2)
        payload = records[0].payload
        self.assertEqual(payload["scheme_id"], SCHEME_ID)
        self.assertEqual(payload["source_url"], "https://files.hdfcfund.com/mid-cap-sid.pdf")
        self.assertEqual(payload["document_title"], "HDFC Mid Cap Fund SID")
        self.assertIn(payload["page_number"], {1, 2})
        self.assertEqual(payload["retrieved_at"], "2026-11-03T00:00:00+00:00")
        self.assertIn(payload["text"], {"Facts for chunk-1.", "Facts for chunk-2."})
        self.assertEqual(len(records[0].vector), 2)
        self.assertEqual(payload["index_generation"], stats.generation)

    def test_same_generation_is_not_duplicated_or_rewritten(self) -> None:
        with patch.object(
            self.client,
            "create_payload_index",
            return_value=True,
        ) as create_payload_index:
            first = index_active_embeddings(
                self.client,
                self.chunk_manifest,
                self.embedding_manifest,
            )
            second = index_active_embeddings(
                self.client,
                self.chunk_manifest,
                self.embedding_manifest,
            )

        self.assertFalse(second.updated)
        self.assertEqual(second.chunks_indexed, 0)
        self.assertEqual(second.collection_name, first.collection_name)
        self.assertEqual(create_payload_index.call_count, 2)
        self.assertEqual(
            create_payload_index.call_args.kwargs["field_schema"],
            models.PayloadSchemaType.KEYWORD,
        )
        self.assertEqual(
            self.client.count(collection_name=ACTIVE_COLLECTION_ALIAS, exact=True).count,
            2,
        )

    def test_changed_generation_switches_alias_and_keeps_old_generation(self) -> None:
        first = index_active_embeddings(
            self.client,
            self.chunk_manifest,
            self.embedding_manifest,
        )
        self.chunk_manifest, self.embedding_manifest = write_index_inputs(
            self.root,
            chunk_ids=("chunk-3",),
            content_hash="b" * 64,
        )

        second = index_active_embeddings(
            self.client,
            self.chunk_manifest,
            self.embedding_manifest,
        )

        alias_target = self.client.get_aliases().aliases[0].collection_name
        self.assertNotEqual(first.generation, second.generation)
        self.assertEqual(alias_target, second.collection_name)
        self.assertEqual(
            self.client.count(collection_name=first.collection_name, exact=True).count,
            2,
        )
        self.assertEqual(
            self.client.count(collection_name=ACTIVE_COLLECTION_ALIAS, exact=True).count,
            1,
        )

    def test_failed_staging_write_does_not_change_active_alias(self) -> None:
        first = index_active_embeddings(
            self.client,
            self.chunk_manifest,
            self.embedding_manifest,
        )
        previous_alias = self.client.get_aliases().aliases[0].collection_name
        self.chunk_manifest, self.embedding_manifest = write_index_inputs(
            self.root,
            chunk_ids=("chunk-3",),
            content_hash="c" * 64,
        )

        with self.assertRaisesRegex(OSError, "simulated Qdrant write failure"):
            index_active_embeddings(
                FailingUpsertClient(self.client),
                self.chunk_manifest,
                self.embedding_manifest,
            )

        self.assertEqual(self.client.get_aliases().aliases[0].collection_name, previous_alias)
        self.assertEqual(
            self.client.count(collection_name=first.collection_name, exact=True).count,
            2,
        )

    def test_failed_alias_update_does_not_remove_previous_alias(self) -> None:
        first = index_active_embeddings(
            self.client,
            self.chunk_manifest,
            self.embedding_manifest,
        )
        self.chunk_manifest, self.embedding_manifest = write_index_inputs(
            self.root,
            chunk_ids=("chunk-3",),
            content_hash="d" * 64,
        )

        with patch.object(
            self.client,
            "update_collection_aliases",
            side_effect=OSError("simulated alias update failure"),
        ):
            with self.assertRaisesRegex(OSError, "alias update failure"):
                index_active_embeddings(
                    self.client,
                    self.chunk_manifest,
                    self.embedding_manifest,
                )

        self.assertEqual(
            self.client.get_aliases().aliases[0].collection_name,
            first.collection_name,
        )
        self.assertEqual(
            self.client.count(collection_name=first.collection_name, exact=True).count,
            2,
        )

    def test_corrupt_embedding_file_is_rejected_before_qdrant_calls(self) -> None:
        manifest = json.loads(self.embedding_manifest.read_text(encoding="utf-8"))
        vector_path = self.embedding_manifest.parent / manifest["documents"][0]["path"]
        vector_path.write_bytes(b"corrupt")
        with patch.object(self.client, "get_aliases") as get_aliases:
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                index_active_embeddings(
                    self.client,
                    self.chunk_manifest,
                    self.embedding_manifest,
                )
        get_aliases.assert_not_called()

    def test_unapproved_source_is_rejected_before_qdrant_calls(self) -> None:
        chunk_file = (
            self.chunk_manifest.parent
            / Path(SCHEME_ID)
            / f"{SOURCE_ID}-{CONTENT_HASH[:12]}.jsonl"
        )
        rows = [json.loads(line) for line in chunk_file.read_text(encoding="utf-8").splitlines()]
        rows[0]["metadata"]["source_url"] = "https://attacker.example/fund.pdf"
        rows[0]["metadata"]["source_domain"] = "attacker.example"
        chunk_file.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        with patch.object(self.client, "get_aliases") as get_aliases:
            with self.assertRaisesRegex(ValueError, "not allowlisted"):
                index_active_embeddings(
                    self.client,
                    self.chunk_manifest,
                    self.embedding_manifest,
                )
        get_aliases.assert_not_called()


if __name__ == "__main__":
    unittest.main()
