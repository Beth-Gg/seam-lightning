"""Pinecone speaker index: namespace = ``project_id``.

Vectors must match the embedding dimension of ``pyannote/embedding`` (512)
and use the same metric your Pinecone index was created with (cosine recommended).
"""
import logging
import os
from typing import Any, Optional, Tuple
from pinecone import Pinecone

logger = logging.getLogger(__name__)


class PineconeSpeakerStore:
    """Query/update speaker vectors scoped by namespace (project_id)."""

    def __init__(self) -> None:
        self._api_key = os.getenv("PINECONE_API_KEY")
        self._index_name = os.getenv("PINECONE_INDEX_NAME")
        self._index: Any = None
        if self._api_key and self._index_name:
            try:

                pc = Pinecone(api_key=self._api_key)
                self._index = pc.Index(self._index_name)
                logger.info("Pinecone index ready: %s", self._index_name)
            except Exception as exc:
                logger.error("Failed to initialize Pinecone: %s", exc)
                self._index = None
        else:
            logger.warning("PINECONE_API_KEY or PINECONE_INDEX_NAME missing; speaker matching disabled.")

    @property
    def enabled(self) -> bool:
        return self._index is not None

    def query_best_speaker(
        self,
        namespace: str,
        vector: list[float],
        top_k: int = 1,
        min_score: float = 0.65,
    ) -> Optional[Tuple[str, float]]:
        """Returns (speaker_label, score) if best match exceeds min_score."""
        if not self._index:
            return None
        try:
            result = self._index.query(
                vector=vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
            )
        except Exception as exc:
            logger.warning("Pinecone query failed for namespace %s: %s", namespace, exc)
            return None

        matches = getattr(result, "matches", None) or []
        if not matches:
            return None
        best = matches[0]
        score = float(getattr(best, "score", 0.0))
        if score < min_score:
            return None
        meta = getattr(best, "metadata", None) or {}
        speaker = (
            meta.get("speaker_label")
            or meta.get("speaker_id")
            or meta.get("user_id")
            or meta.get("label")
            or str(getattr(best, "id", ""))
        )
        if not speaker:
            return None
        return speaker, score

    def upsert_speaker(
        self,
        namespace: str,
        vector_id: str,
        vector: list[float],
        metadata: dict[str, Any],
    ) -> None:
        if not self._index:
            raise RuntimeError("Pinecone is not configured.")
        self._index.upsert(vectors=[{"id": vector_id, "values": vector, "metadata": metadata}], namespace=namespace)
