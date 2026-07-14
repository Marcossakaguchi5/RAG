import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from qdrant_client import models


API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))

from app.services import indexer  # noqa: E402


class IndexDocumentChunksTests(unittest.TestCase):
    def test_indexes_and_upserts_each_configured_batch(self):
        chunks = [
            SimpleNamespace(id=f"chunk-{index}", content=f"content-{index}")
            for index in range(5)
        ]
        document = SimpleNamespace(collection_name="test-collection")
        dense = SimpleNamespace(encode=lambda texts: [[float(len(text))] for text in texts])
        sparse = SimpleNamespace(
            encode_documents=lambda texts: [
                models.SparseVector(indices=[index], values=[1.0])
                for index, _ in enumerate(texts)
            ]
        )
        upsert_calls = []

        with (
            patch.object(indexer, "get_settings", return_value=SimpleNamespace(index_batch_size=2)),
            patch.object(indexer, "get_embedding_service", return_value=dense),
            patch.object(indexer, "get_sparse_embedding_service", return_value=sparse),
            patch.object(indexer, "_chunk_payload", return_value={}),
            patch.object(
                indexer,
                "upsert_chunks",
                side_effect=lambda points, collection: upsert_calls.append((points, collection)),
            ),
        ):
            indexer.index_document_chunks(document, chunks)

        self.assertEqual([len(points) for points, _ in upsert_calls], [2, 2, 1])
        self.assertEqual({collection for _, collection in upsert_calls}, {"test-collection"})
        self.assertEqual(
            [point.id for points, _ in upsert_calls for point in points],
            [chunk.id for chunk in chunks],
        )


if __name__ == "__main__":
    unittest.main()
