"""Persistent, atomic API response caching for FactIndic.

Provides a thread-safe and process-safe disk cache backed by SQLite to eliminate
redundant LLM generation, claim extraction, and NLI verification calls.
Supports deterministic hashing, cache hit/miss statistics, and offline dry-runs.
"""

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def compute_cache_key(namespace: str, payload: Dict[str, Any]) -> str:
    """Generate a deterministic SHA-256 cache key from a namespace and payload dictionary.

    Args:
        namespace: Cache namespace ('nli', 'claims', 'generation').
        payload: Dictionary of input parameters defining the request.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    hasher = hashlib.sha256()
    hasher.update(namespace.encode("utf-8"))
    hasher.update(b":")
    hasher.update(serialized.encode("utf-8"))
    return hasher.hexdigest()


@dataclass
class CacheStats:
    """Statistics for cache operations."""
    hits: int = 0
    misses: int = 0
    writes: int = 0
    total_entries: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "total_entries": self.total_entries,
        }


class ResponseCache:
    """Persistent SQLite-backed cache for API responses."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize ResponseCache.

        Args:
            db_path: Path to SQLite database file. If None, defaults to
                'data/cache/response_cache.db'. Use ':memory:' for isolated testing.
        """
        if db_path is None:
            self.db_path = Path(__file__).resolve().parent.parent / "data" / "cache" / "response_cache.db"
        elif str(db_path) == ":memory:":
            self.db_path = Path(":memory:")
        else:
            self.db_path = Path(db_path)

        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection with timeout and WAL mode enabled for file databases."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        if str(self.db_path) != ":memory:":
            conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        """Create database tables and indexes if they do not exist."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS response_cache (
                    cache_key TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    metadata_json TEXT,
                    created_at_utc TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_cache_namespace
                ON response_cache(namespace);
                """
            )

    def get(self, namespace: str, payload: Dict[str, Any]) -> Optional[Any]:
        """Retrieve a cached response if present.

        Args:
            namespace: Namespace ('nli', 'claims', 'generation').
            payload: Input payload parameters.

        Returns:
            Deserialized cached response object, or None on cache miss.
        """
        key = compute_cache_key(namespace, payload)
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT response_json FROM response_cache WHERE cache_key = ?",
                (key,),
            )
            row = cursor.fetchone()
            if row:
                self._hits += 1
                return json.loads(row["response_json"])

        self._misses += 1
        return None

    def set(
        self,
        namespace: str,
        payload: Dict[str, Any],
        response_data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a response in the cache atomically.

        Args:
            namespace: Namespace ('nli', 'claims', 'generation').
            payload: Input payload parameters.
            response_data: Serializable data to cache.
            metadata: Optional metadata for provenance tracking.

        Returns:
            The computed cache key.
        """
        key = compute_cache_key(namespace, payload)
        now_utc = datetime.now(timezone.utc).isoformat()
        resp_json = json.dumps(response_data, ensure_ascii=False)
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO response_cache
                (cache_key, namespace, response_json, metadata_json, created_at_utc)
                VALUES (?, ?, ?, ?, ?);
                """,
                (key, namespace, resp_json, meta_json, now_utc),
            )
        self._writes += 1
        return key

    def has(self, namespace: str, payload: Dict[str, Any]) -> bool:
        """Check if an item exists in the cache without updating hit/miss counters."""
        key = compute_cache_key(namespace, payload)
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM response_cache WHERE cache_key = ?",
                (key,),
            )
            return cursor.fetchone() is not None

    # -----------------------------------------------------------------------
    # Convenience methods for NLI inference
    # -----------------------------------------------------------------------

    def get_nli(
        self,
        premise: str,
        hypothesis: str,
        language: str,
        model_id: str,
        provider: str = "default",
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached NLI result for a premise-hypothesis pair."""
        payload = {
            "premise": premise.strip(),
            "hypothesis": hypothesis.strip(),
            "language": language.strip().lower(),
            "model_id": model_id.strip(),
            "provider": provider.strip().lower(),
        }
        return self.get("nli", payload)

    def set_nli(
        self,
        premise: str,
        hypothesis: str,
        language: str,
        model_id: str,
        result_dict: Dict[str, Any],
        provider: str = "default",
    ) -> str:
        """Store an NLI evaluation result in cache."""
        payload = {
            "premise": premise.strip(),
            "hypothesis": hypothesis.strip(),
            "language": language.strip().lower(),
            "model_id": model_id.strip(),
            "provider": provider.strip().lower(),
        }
        return self.set("nli", payload, result_dict)

    # -----------------------------------------------------------------------
    # Convenience methods for atomic claim extraction
    # -----------------------------------------------------------------------

    def get_claims(
        self,
        sentence: str,
        language: str,
        model_id: str,
        provider: str = "default",
    ) -> Optional[List[str]]:
        """Retrieve cached atomic claims for a summary sentence."""
        payload = {
            "sentence": sentence.strip(),
            "language": language.strip().lower(),
            "model_id": model_id.strip(),
            "provider": provider.strip().lower(),
        }
        return self.get("claims", payload)

    def set_claims(
        self,
        sentence: str,
        language: str,
        model_id: str,
        claims: List[str],
        provider: str = "default",
    ) -> str:
        """Store extracted atomic claims in cache."""
        payload = {
            "sentence": sentence.strip(),
            "language": language.strip().lower(),
            "model_id": model_id.strip(),
            "provider": provider.strip().lower(),
        }
        return self.set("claims", payload, claims)

    # -----------------------------------------------------------------------
    # Convenience methods for summary generation
    # -----------------------------------------------------------------------

    def get_generation(
        self,
        source_text: str,
        language: str,
        prompt_strategy: str,
        model_id: str,
        provider: str = "default",
    ) -> Optional[str]:
        """Retrieve cached generated summary for a source article."""
        payload = {
            "source_text": source_text.strip(),
            "language": language.strip().lower(),
            "prompt_strategy": prompt_strategy.strip().lower(),
            "model_id": model_id.strip(),
            "provider": provider.strip().lower(),
        }
        return self.get("generation", payload)

    def set_generation(
        self,
        source_text: str,
        language: str,
        prompt_strategy: str,
        model_id: str,
        summary: str,
        provider: str = "default",
    ) -> str:
        """Store a generated summary in cache."""
        payload = {
            "source_text": source_text.strip(),
            "language": language.strip().lower(),
            "prompt_strategy": prompt_strategy.strip().lower(),
            "model_id": model_id.strip(),
            "provider": provider.strip().lower(),
        }
        return self.set("generation", payload, summary)

    # -----------------------------------------------------------------------
    # Statistics and Management
    # -----------------------------------------------------------------------

    def get_stats(self) -> CacheStats:
        """Get cache performance statistics and total entry count."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) AS total FROM response_cache")
            total = cursor.fetchone()["total"]
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            writes=self._writes,
            total_entries=total,
        )

    def clear(self, namespace: Optional[str] = None) -> int:
        """Clear cache entries, optionally scoped to a namespace.

        Returns:
            Number of rows deleted.
        """
        with self._get_connection() as conn:
            if namespace:
                cursor = conn.execute(
                    "DELETE FROM response_cache WHERE namespace = ?",
                    (namespace,),
                )
            else:
                cursor = conn.execute("DELETE FROM response_cache")
            return cursor.rowcount
