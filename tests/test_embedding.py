import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from kuvera_lens.embedding import embed_active_chunks


class FakeEmbeddingModel:
    def __init__(self, *, invalid: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.invalid = invalid

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        self.calls.append(sentences)
        if self.invalid:
            return np.zeros((len(sentences), 2), dtype=np.float32)
        return np.tile(
            np.asarray([[0.6, 0.8]], dtype=np.float32),
            (len(sentences), 1),
        )


def write_chunk_set(
    root: Path,
    *,
    content_hash: str = "a" * 64,
    chunk_ids: tuple[str, ...] = ("chunk-1", "chunk-2"),
) -> Path:
    source_id = "mid-cap-sid"
    scheme_id = "hdfc_mid_cap_direct_growth"
    relative_file = Path(scheme_id) / f"{source_id}-{content_hash[:12]}.jsonl"
    chunk_file = root / relative_file
    chunk_file.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "chunk_id": chunk_id,
            "text": f"Facts for {chunk_id}.",
            "metadata": {
                "source_id": source_id,
                "scheme_id": scheme_id,
                "content_hash": content_hash,
                "page_number": index + 1,
                "source_url": "https://files.hdfcfund.com/mid-cap-sid.pdf",
            },
        }
        for index, chunk_id in enumerate(chunk_ids)
    ]
    chunk_file.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "documents": 1,
        "chunks": len(rows),
        "files": [
            {
                "source_id": source_id,
                "scheme_id": scheme_id,
                "content_hash": content_hash,
                "chunk_count": len(rows),
                "path": relative_file.as_posix(),
            }
        ],
    }
    manifest_path = root / "active_chunks.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


class EmbeddingTests(unittest.TestCase):
    def test_embeds_active_chunks_as_normalized_aligned_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            chunk_manifest = write_chunk_set(root / "chunks")
            model = FakeEmbeddingModel()

            result = embed_active_chunks(
                chunk_manifest,
                root / "embeddings",
                model=model,
                batch_size=1,
            )

            active = json.loads(result.active_manifest.read_text(encoding="utf-8"))
            document = active["documents"][0]
            vector_path = root / "embeddings" / document["path"]
            with np.load(vector_path, allow_pickle=False) as vectors:
                self.assertEqual(vectors["chunk_ids"].tolist(), ["chunk-1", "chunk-2"])
                self.assertEqual(vectors["embeddings"].shape, (2, 2))
                np.testing.assert_allclose(
                    np.linalg.norm(vectors["embeddings"], axis=1),
                    np.ones(2),
                )

        self.assertEqual(result.documents_embedded, 1)
        self.assertEqual(result.chunks_embedded, 2)
        self.assertEqual(result.unchanged_documents, 0)
        self.assertEqual(result.dimensions, 2)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(document["chunk_file"], "hdfc_mid_cap_direct_growth/mid-cap-sid-aaaaaaaaaaaa.jsonl")
        self.assertEqual(document["scheme_id"], "hdfc_mid_cap_direct_growth")

    def test_skips_unchanged_embedding_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            chunk_manifest = write_chunk_set(root / "chunks")
            output = root / "embeddings"
            embed_active_chunks(chunk_manifest, output, model=FakeEmbeddingModel())
            active_before = (output / "active_embeddings.json").read_bytes()
            second_model = FakeEmbeddingModel()

            result = embed_active_chunks(
                chunk_manifest,
                output,
                model=second_model,
            )

            self.assertEqual(result.documents_embedded, 0)
            self.assertEqual(result.chunks_embedded, 0)
            self.assertEqual(result.unchanged_documents, 1)
            self.assertFalse(result.updated)
            self.assertEqual(second_model.calls, [])
            self.assertEqual(
                (output / "active_embeddings.json").read_bytes(),
                active_before,
            )

    def test_publishes_manifest_without_retired_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            chunk_manifest = write_chunk_set(root / "chunks")
            output = root / "embeddings"
            first = embed_active_chunks(
                chunk_manifest,
                output,
                model=FakeEmbeddingModel(),
            )
            previous = json.loads(first.active_manifest.read_text(encoding="utf-8"))
            previous["documents"].append(
                {
                    "source_id": "retired-sid",
                    "scheme_id": "hdfc_mid_cap_direct_growth",
                }
            )
            first.active_manifest.write_text(json.dumps(previous), encoding="utf-8")

            result = embed_active_chunks(
                chunk_manifest,
                output,
                model=FakeEmbeddingModel(),
            )

            active = json.loads(result.active_manifest.read_text(encoding="utf-8"))
            self.assertTrue(result.updated)
            self.assertEqual(result.documents_embedded, 0)
            self.assertEqual(
                [item["source_id"] for item in active["documents"]],
                ["mid-cap-sid"],
            )

    def test_changed_chunks_get_new_vectors_and_publish_new_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            chunk_directory = root / "chunks"
            output = root / "embeddings"
            chunk_manifest = write_chunk_set(chunk_directory)
            first = embed_active_chunks(
                chunk_manifest, output, model=FakeEmbeddingModel()
            )
            previous_active = first.active_manifest.read_bytes()
            write_chunk_set(
                chunk_directory,
                content_hash="b" * 64,
                chunk_ids=("chunk-3",),
            )

            model = FakeEmbeddingModel()
            result = embed_active_chunks(chunk_manifest, output, model=model)

            active = json.loads(result.active_manifest.read_text(encoding="utf-8"))
            self.assertEqual(result.documents_embedded, 1)
            self.assertEqual(result.chunks_embedded, 1)
            self.assertEqual(active["chunks"], 1)
            self.assertNotEqual(result.active_manifest.read_bytes(), previous_active)
            self.assertEqual(model.calls, [["Facts for chunk-3."]])

    def test_rejects_non_normalized_vectors_without_replacing_active_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            chunk_directory = root / "chunks"
            output = root / "embeddings"
            chunk_manifest = write_chunk_set(chunk_directory)
            result = embed_active_chunks(
                chunk_manifest, output, model=FakeEmbeddingModel()
            )
            previous_active = result.active_manifest.read_bytes()
            write_chunk_set(
                chunk_directory,
                content_hash="b" * 64,
                chunk_ids=("chunk-3",),
            )

            with self.assertRaisesRegex(ValueError, "non-normalized"):
                embed_active_chunks(
                    chunk_manifest,
                    output,
                    model=FakeEmbeddingModel(invalid=True),
                )

            self.assertEqual(result.active_manifest.read_bytes(), previous_active)

    def test_manifest_publication_failure_preserves_previous_active_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            chunk_directory = root / "chunks"
            output = root / "embeddings"
            chunk_manifest = write_chunk_set(chunk_directory)
            result = embed_active_chunks(
                chunk_manifest, output, model=FakeEmbeddingModel()
            )
            previous_active = result.active_manifest.read_bytes()
            write_chunk_set(
                chunk_directory,
                content_hash="c" * 64,
                chunk_ids=("chunk-4",),
            )

            with patch(
                "kuvera_lens.embedding._publish_manifest",
                side_effect=OSError("simulated publication failure"),
            ):
                with self.assertRaisesRegex(OSError, "publication failure"):
                    embed_active_chunks(
                        chunk_manifest,
                        output,
                        model=FakeEmbeddingModel(),
                    )

            self.assertEqual(result.active_manifest.read_bytes(), previous_active)

    def test_invalid_chunk_manifest_fails_before_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            chunk_manifest = write_chunk_set(root / "chunks")
            manifest = json.loads(chunk_manifest.read_text(encoding="utf-8"))
            manifest["chunks"] = 99
            chunk_manifest.write_text(json.dumps(manifest), encoding="utf-8")
            model = FakeEmbeddingModel()

            with self.assertRaisesRegex(ValueError, "total mismatch"):
                embed_active_chunks(
                    chunk_manifest,
                    root / "embeddings",
                    model=model,
                )

            self.assertEqual(model.calls, [])


if __name__ == "__main__":
    unittest.main()
