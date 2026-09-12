"""Build and query a rebuildable, read-only-derived index for Ombre buckets.

The source of truth remains the Markdown files. This module only writes the
separate SQLite sidecar passed through --index-db.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import jieba  # type: ignore
except ImportError:  # pragma: no cover - optional local fallback
    jieba = None


LOGGER = logging.getLogger("ombre-recall")
SCHEMA_VERSION = "1"
DEFAULT_CHUNK_CHARS = 900
RRF_K = 60


@dataclass(frozen=True)
class MemoryDocument:
    doc_id: str
    path: str
    kind: str
    name: str
    domains: tuple[str, ...]
    tags: tuple[str, ...]
    importance: int
    valence: float
    arousal: float
    created: str
    resolved: bool
    body: str
    sha256: str
    mtime_ns: int


def _as_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _as_float(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _normalize_tokens(text: str) -> list[str]:
    clean = str(text or "").lower().replace("\u3000", " ")
    if jieba is not None:
        pieces = jieba.lcut(clean, cut_all=False)
    else:
        pieces = re.findall(r"[a-z0-9_./:@-]+|[\u4e00-\u9fff]+", clean)

    tokens: list[str] = []
    for piece in pieces:
        token = re.sub(r"\s+", "", str(piece))
        if not token or token in {"-", "_"}:
            continue
        if len(token) == 1 and not re.match(r"[\u4e00-\u9fff]", token):
            continue
        tokens.append(token)
    return tokens


def fts_text(text: str) -> str:
    return " ".join(_normalize_tokens(text))


def _split_body(body: str, max_chars: int = DEFAULT_CHUNK_CHARS) -> list[str]:
    """Split at paragraph boundaries while keeping each bucket recognizable."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", body or "") if part.strip()]
    if not paragraphs:
        return [""]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars and len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}".strip()
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(paragraph) <= max_chars:
            current = paragraph
            continue
        for offset in range(0, len(paragraph), max_chars):
            chunks.append(paragraph[offset : offset + max_chars].strip())

    if current:
        chunks.append(current)
    return chunks or [body[:max_chars]]


def _relative_kind(path: Path, root: Path, metadata: dict[str, Any]) -> str:
    try:
        relative = path.relative_to(root)
        root_name = relative.parts[0] if relative.parts else "dynamic"
    except ValueError:
        root_name = "dynamic"
    # Storage class is determined by the directory, not by frontmatter. Ombre
    # marks archived files as "archived" in metadata, while the query filter
    # needs one stable value for every file under buckets/archive/.
    if root_name in {"permanent", "dynamic", "archive"}:
        return root_name
    return str(metadata.get("type") or root_name)


def load_document(path: Path, root: Path) -> MemoryDocument:
    try:
        import frontmatter  # type: ignore
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("python-frontmatter is required to parse Ombre buckets") from exc

    post = frontmatter.load(str(path))
    metadata = dict(post.metadata or {})
    body = str(post.content or "").strip()
    raw = path.read_bytes()
    stat = path.stat()
    doc_id = str(metadata.get("id") or path.stem)
    name = str(metadata.get("name") or path.stem)
    domains = tuple(_as_list(metadata.get("domain")))
    tags = tuple(_as_list(metadata.get("tags")))
    return MemoryDocument(
        doc_id=doc_id,
        path=str(path),
        kind=_relative_kind(path, root, metadata),
        name=name,
        domains=domains,
        tags=tags,
        importance=max(1, min(10, _as_int(metadata.get("importance"), 5))),
        valence=_as_float(metadata.get("valence"), 0.5),
        arousal=_as_float(metadata.get("arousal"), 0.3),
        created=str(metadata.get("created") or ""),
        resolved=bool(metadata.get("resolved", False)),
        body=body,
        sha256=hashlib.sha256(raw).hexdigest(),
        mtime_ns=stat.st_mtime_ns,
    )


def scan_documents(buckets_dir: str | Path, include_archive: bool = False) -> list[MemoryDocument]:
    root = Path(buckets_dir).resolve()
    scan_roots = [root / "permanent", root / "dynamic"]
    if include_archive:
        scan_roots.append(root / "archive")

    documents: list[MemoryDocument] = []
    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        for path in sorted(scan_root.rglob("*.md")):
            try:
                documents.append(load_document(path, root))
            except Exception as exc:
                LOGGER.warning("skip unreadable bucket %s: %s", path, exc)
    return documents


def _vector_blob(vector: Sequence[float]) -> bytes:
    values = [float(item) for item in vector]
    return struct.pack(f"<{len(values)}f", *values)


def _vector_from_blob(blob: bytes, dims: int) -> tuple[float, ...]:
    return struct.unpack(f"<{dims}f", blob)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class FastEmbedder:
    """Optional local semantic backend; importing it is deferred."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding  # type: ignore

        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name)

    def embed(self, texts: Iterable[str]) -> list[tuple[float, ...]]:
        return [tuple(float(value) for value in vector) for vector in self.model.embed(list(texts))]


def init_db(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS documents (
          doc_id TEXT PRIMARY KEY,
          path TEXT NOT NULL,
          kind TEXT NOT NULL,
          name TEXT NOT NULL,
          domains_json TEXT NOT NULL,
          tags_json TEXT NOT NULL,
          importance INTEGER NOT NULL,
          valence REAL NOT NULL,
          arousal REAL NOT NULL,
          created TEXT NOT NULL,
          resolved INTEGER NOT NULL,
          sha256 TEXT NOT NULL,
          mtime_ns INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
          chunk_id TEXT PRIMARY KEY,
          doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
          ordinal INTEGER NOT NULL,
          text TEXT NOT NULL,
          fts_text TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
          chunk_id UNINDEXED,
          doc_id UNINDEXED,
          fts_text
        );
        CREATE TABLE IF NOT EXISTS embeddings (
          chunk_id TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
          model TEXT NOT NULL,
          dims INTEGER NOT NULL,
          vector BLOB NOT NULL
        );
        """
    )


def _embedding_text(document: MemoryDocument, chunk: str) -> str:
    metadata = " ".join([document.name, *document.domains, *document.tags])
    return f"{metadata}\n{chunk}".strip()


def _source_fingerprint(documents: Sequence[MemoryDocument]) -> str:
    digest = hashlib.sha256()
    for document in sorted(documents, key=lambda item: (item.kind, item.doc_id, item.path)):
        digest.update(
            f"{document.kind}\0{document.doc_id}\0{document.sha256}\n".encode("utf-8")
        )
    return digest.hexdigest()


def rebuild_index(
    buckets_dir: str | Path,
    index_db: str | Path,
    include_archive: bool = False,
    embedder: FastEmbedder | None = None,
) -> dict[str, int | str]:
    """Rebuild only the sidecar database; never opens source files for writing."""
    documents = scan_documents(buckets_dir, include_archive=include_archive)
    db_path = Path(index_db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    source_fingerprint = _source_fingerprint(documents)
    embedding_model = embedder.model_name if embedder else ""

    # Avoid repeatedly embedding unchanged memories when the systemd timer
    # runs. The source fingerprint is derived only from file identity + hash.
    if db_path.exists():
        try:
            existing = sqlite3.connect(str(db_path))
            meta = dict(existing.execute("SELECT key, value FROM meta").fetchall())
            existing.close()
            if (
                meta.get("schema_version") == SCHEMA_VERSION
                and meta.get("source_fingerprint") == source_fingerprint
                and meta.get("include_archive") == str(bool(include_archive))
                and meta.get("embedding_model", "") == embedding_model
            ):
                return {
                    "documents": len(documents),
                    "chunks": sum(len(_split_body(document.body)) for document in documents),
                    "embedding_model": embedding_model,
                    "skipped": 1,
                }
        except Exception as exc:
            LOGGER.info("sidecar metadata unavailable; rebuilding: %s", exc)

    # Build beside the live file and atomically replace it after closing the
    # connection. Readers either see the old complete index or the new one.
    temp_db_path = db_path.with_name(f".{db_path.name}.tmp-{os.getpid()}")
    connection = sqlite3.connect(str(temp_db_path))
    try:
        init_db(connection)
        with connection:
            connection.execute("DELETE FROM embeddings")
            connection.execute("DELETE FROM chunks_fts")
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM documents")

            for document in documents:
                connection.execute(
                    """
                    INSERT INTO documents
                      (doc_id, path, kind, name, domains_json, tags_json,
                       importance, valence, arousal, created, resolved, sha256, mtime_ns)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document.doc_id,
                        document.path,
                        document.kind,
                        document.name,
                        json.dumps(document.domains, ensure_ascii=False),
                        json.dumps(document.tags, ensure_ascii=False),
                        document.importance,
                        document.valence,
                        document.arousal,
                        document.created,
                        int(document.resolved),
                        document.sha256,
                        document.mtime_ns,
                    ),
                )

                chunks = _split_body(document.body)
                for ordinal, chunk in enumerate(chunks):
                    chunk_id = f"{document.doc_id}:{ordinal}"
                    search_text = _embedding_text(document, chunk)
                    tokenized = fts_text(search_text)
                    connection.execute(
                        "INSERT INTO chunks(chunk_id, doc_id, ordinal, text, fts_text) VALUES (?, ?, ?, ?, ?)",
                        (chunk_id, document.doc_id, ordinal, chunk, tokenized),
                    )
                    connection.execute(
                        "INSERT INTO chunks_fts(chunk_id, doc_id, fts_text) VALUES (?, ?, ?)",
                        (chunk_id, document.doc_id, tokenized),
                    )

                    if embedder is not None:
                        vectors = embedder.embed([search_text])
                        vector = vectors[0]
                        connection.execute(
                            "INSERT INTO embeddings(chunk_id, model, dims, vector) VALUES (?, ?, ?, ?)",
                            (chunk_id, embedder.model_name, len(vector), _vector_blob(vector)),
                        )

            connection.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)", (SCHEMA_VERSION,))
            connection.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('source_dir', ?)", (str(Path(buckets_dir).resolve()),))
            connection.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('include_archive', ?)", (str(bool(include_archive)),))
            connection.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('embedding_model', ?)", (embedding_model,))
            connection.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('source_fingerprint', ?)", (source_fingerprint,))
    finally:
        connection.close()

    os.chmod(temp_db_path, 0o600)
    os.replace(temp_db_path, db_path)

    return {
        "documents": len(documents),
        "chunks": sum(len(_split_body(document.body)) for document in documents),
        "embedding_model": embedding_model,
        "skipped": 0,
    }


def _fts_query(query: str) -> str:
    tokens = _normalize_tokens(query)
    if not tokens:
        return ""
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens[:32])


def _domain_matches(domains_json: str, domains: Sequence[str] | None) -> bool:
    if not domains:
        return True
    current = {str(item).strip().lower() for item in json.loads(domains_json or "[]")}
    wanted = {str(item).strip().lower() for item in domains if str(item).strip()}
    return not wanted or bool(current & wanted)


def _recency_score(created: str) -> float:
    if not created:
        return 0.5
    try:
        date_text = created.replace("Z", "+00:00")
        stamp = datetime.fromisoformat(date_text)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        days = max(0.0, (datetime.now(timezone.utc) - stamp).total_seconds() / 86400)
        return math.exp(-0.02 * days)
    except ValueError:
        return 0.5


def search_index(
    index_db: str | Path,
    query: str,
    limit: int = 5,
    domains: Sequence[str] | None = None,
    include_archive: bool = False,
    embedder: FastEmbedder | None = None,
) -> list[dict[str, Any]]:
    """Search the sidecar without touching source files or activation metadata."""
    query = str(query or "").strip()
    if not query:
        return []

    connection = sqlite3.connect(str(index_db))
    connection.row_factory = sqlite3.Row
    try:
        init_db(connection)
        lexical: dict[str, int] = {}
        lexical_chunk: dict[str, str] = {}
        match = _fts_query(query)
        if match:
            # FTS5 does not allow bm25() in the aggregate expression used by
            # the old version of this query. Rank matching chunks first, then
            # keep the first occurrence for each document in Python.
            rows = connection.execute(
                """
                SELECT chunks_fts.doc_id AS doc_id, chunks_fts.chunk_id AS chunk_id, bm25(chunks_fts) AS rank_score
                FROM chunks_fts
                JOIN documents ON documents.doc_id = chunks_fts.doc_id
                WHERE chunks_fts MATCH ?
                  AND (? OR documents.kind != 'archive')
                ORDER BY rank_score ASC
                LIMIT 100
                """,
                (match, int(include_archive)),
            ).fetchall()
            for row in rows:
                doc_id = str(row["doc_id"])
                if doc_id not in lexical:
                    lexical[doc_id] = len(lexical) + 1
                    lexical_chunk[doc_id] = str(row["chunk_id"])

        semantic: dict[str, tuple[int, float]] = {}
        semantic_chunk: dict[str, str] = {}
        if embedder is not None:
            try:
                query_vector = embedder.embed([query])[0]
                rows = connection.execute(
                    """
                    SELECT e.chunk_id, e.dims, e.vector, c.doc_id, d.kind
                    FROM embeddings e
                    JOIN chunks c ON c.chunk_id = e.chunk_id
                    JOIN documents d ON d.doc_id = c.doc_id
                    WHERE (? OR d.kind != 'archive')
                    """,
                    (int(include_archive),),
                ).fetchall()
                scored: list[tuple[str, str, float]] = []
                for row in rows:
                    vector = _vector_from_blob(row["vector"], int(row["dims"]))
                    scored.append((str(row["doc_id"]), str(row["chunk_id"]), cosine_similarity(query_vector, vector)))
                scored.sort(key=lambda item: item[2], reverse=True)
                best_by_doc: dict[str, float] = {}
                for doc_id, chunk_id, score in scored:
                    if score > best_by_doc.get(doc_id, -1.0):
                        best_by_doc[doc_id] = score
                        semantic_chunk[doc_id] = chunk_id
                for rank, (doc_id, score) in enumerate(sorted(best_by_doc.items(), key=lambda item: item[1], reverse=True)[:50], start=1):
                    semantic[doc_id] = (rank, score)
            except Exception as exc:  # lexical search remains available
                LOGGER.warning("semantic search unavailable: %s", exc)

        candidate_ids = set(lexical) | set(semantic)
        if not candidate_ids:
            return []

        placeholders = ",".join("?" for _ in candidate_ids)
        rows = connection.execute(
            f"SELECT * FROM documents WHERE doc_id IN ({placeholders})",
            tuple(candidate_ids),
        ).fetchall()

        scored_docs: list[tuple[float, sqlite3.Row, dict[str, Any]]] = []
        for row in rows:
            if not _domain_matches(row["domains_json"], domains):
                continue
            doc_id = str(row["doc_id"])
            rrf = 0.0
            if doc_id in lexical:
                rrf += 1.0 / (RRF_K + lexical[doc_id])
            if doc_id in semantic:
                rrf += 1.0 / (RRF_K + semantic[doc_id][0])
            payload = {
                "id": doc_id,
                "name": row["name"],
                "kind": row["kind"],
                "domains": json.loads(row["domains_json"] or "[]"),
                "tags": json.loads(row["tags_json"] or "[]"),
                "importance": int(row["importance"]),
                "created": row["created"],
                "resolved": bool(row["resolved"]),
                "evidence": {
                    "lexical_rank": lexical.get(doc_id),
                    "semantic_rank": semantic.get(doc_id, (None, None))[0],
                    "semantic_similarity": semantic.get(doc_id, (None, None))[1],
                },
            }
            scored_docs.append((rrf, row, payload))

        if not scored_docs:
            return []
        max_rrf = max(item[0] for item in scored_docs) or 1.0
        final: list[dict[str, Any]] = []
        for rrf, row, payload in scored_docs:
            base = rrf / max_rrf
            importance_prior = int(row["importance"]) / 10.0
            recency_prior = _recency_score(str(row["created"] or ""))
            final_score = 0.72 * base + 0.16 * importance_prior + 0.12 * recency_prior
            best_chunk_id = semantic_chunk.get(str(row["doc_id"])) or lexical_chunk.get(str(row["doc_id"]))
            if best_chunk_id:
                chunk = connection.execute(
                    "SELECT text FROM chunks WHERE chunk_id = ?",
                    (best_chunk_id,),
                ).fetchone()
            else:
                chunk = connection.execute(
                    "SELECT text FROM chunks WHERE doc_id = ? ORDER BY ordinal ASC LIMIT 1",
                    (row["doc_id"],),
                ).fetchone()
            payload["score"] = round(final_score, 6)
            payload["excerpt"] = str(chunk["text"] if chunk else "")[:800]
            final.append(payload)

        final.sort(key=lambda item: item["score"], reverse=True)
        return final[: max(1, min(int(limit), 20))]
    finally:
        connection.close()


def _build_embedder(model_name: str | None, disabled: bool) -> FastEmbedder | None:
    if disabled or not model_name:
        return None
    try:
        return FastEmbedder(model_name)
    except Exception as exc:
        LOGGER.warning("embedding backend unavailable; continue lexical-only: %s", exc)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Ombre read-only recall sidecar")
    parser.add_argument("--buckets-dir", required=True)
    parser.add_argument("--index-db", required=True)
    parser.add_argument("--include-archive", action="store_true")
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--embedding-model", default=os.environ.get("RECALL_EMBEDDING_MODEL", ""))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    embedder = _build_embedder(args.embedding_model, args.no_embeddings)
    result = rebuild_index(args.buckets_dir, args.index_db, args.include_archive, embedder)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
