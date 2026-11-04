import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kuvera_lens.config import Settings
from kuvera_lens.qdrant_store import ACTIVE_COLLECTION_ALIAS, create_qdrant_client
from kuvera_lens.retrieval import get_retrieval_client


class SettingsTests(unittest.TestCase):
    def test_reads_configuration_from_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "GROQ_API_KEY": "test-key",
                "QDRANT_URL": "https://example.qdrant.io:6333",
                "QDRANT_API_KEY": "test-qdrant-key",
            },
            clear=True,
        ):
            settings = Settings.from_environment()

        self.assertEqual(settings.groq_api_key, "test-key")
        self.assertEqual(settings.qdrant_url, "https://example.qdrant.io:6333")
        self.assertEqual(settings.qdrant_api_key, "test-qdrant-key")

    def test_requires_groq_key_when_answer_generation_needs_it(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_environment()

        with self.assertRaisesRegex(RuntimeError, "GROQ_API_KEY is required"):
            settings.require_groq_api_key()

    def test_requires_complete_qdrant_cloud_configuration(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_environment()

        with self.assertRaisesRegex(
            RuntimeError,
            "QDRANT_URL, QDRANT_API_KEY",
        ):
            settings.require_qdrant_cloud_config()

    def test_returns_qdrant_cloud_configuration(self) -> None:
        with patch.dict(
            os.environ,
            {
                "QDRANT_URL": "https://example.qdrant.io:6333",
                "QDRANT_API_KEY": "test-qdrant-key",
            },
            clear=True,
        ):
            settings = Settings.from_environment()

        self.assertEqual(
            settings.require_qdrant_cloud_config(),
            ("https://example.qdrant.io:6333", "test-qdrant-key"),
        )

    def test_local_client_uses_only_the_separate_local_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            local_path = os.path.join(temporary_directory, "dev-qdrant")
            with patch.dict(os.environ, {}, clear=True):
                settings = Settings.from_environment()
            client = create_qdrant_client(settings, local_path=Path(local_path))
            try:
                self.assertTrue(os.path.isdir(local_path))
                self.assertEqual(ACTIVE_COLLECTION_ALIAS, "kuvera_lens_active")
            finally:
                client.close()

    def test_partial_cloud_settings_do_not_fall_back_to_local(self) -> None:
        with patch.dict(
            os.environ,
            {"QDRANT_URL": "https://example.qdrant.io:6333"},
            clear=True,
        ):
            settings = Settings.from_environment()

        with self.assertRaisesRegex(RuntimeError, "QDRANT_API_KEY"):
            create_qdrant_client(settings)

    def test_cloud_client_uses_the_configured_shared_endpoint(self) -> None:
        settings = Settings(
            groq_api_key=None,
            qdrant_url="https://shared.qdrant.cloud:6333",
            qdrant_api_key="test-key",
        )
        with patch("kuvera_lens.qdrant_store.QdrantClient") as qdrant_client:
            create_qdrant_client(settings)

        qdrant_client.assert_called_once_with(
            url="https://shared.qdrant.cloud:6333",
            api_key="test-key",
        )

    def test_retrieval_uses_the_shared_qdrant_client_factory(self) -> None:
        expected_client = object()
        with patch(
            "kuvera_lens.retrieval.create_qdrant_client",
            return_value=expected_client,
        ) as factory:
            client = get_retrieval_client()

        self.assertIs(client, expected_client)
        factory.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
