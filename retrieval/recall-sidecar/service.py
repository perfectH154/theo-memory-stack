"""Loopback-only HTTP service for read-only Ombre recall candidates."""

from __future__ import annotations

import argparse
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from indexer import FastEmbedder, fts_text, search_index


LOGGER = logging.getLogger("ombre-recall-service")


class RecallServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, index_db, embedder):
        super().__init__(address, handler)
        self.index_db = index_db
        self.embedder = embedder


class Handler(BaseHTTPRequestHandler):
    server: RecallServer

    def log_message(self, format: str, *args):  # noqa: A003 - stdlib hook name
        LOGGER.info("%s - %s", self.address_string(), format % args)

    def _send(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - stdlib hook name
        parsed = urlparse(self.path)
        if parsed.path in {"/healthz", "/health"}:
            self._send(200, {"ok": True, "read_only": True, "service": "ombre-recall-sidecar"})
            return
        if parsed.path != "/search":
            self._send(404, {"ok": False, "error": "not_found"})
            return

        params = parse_qs(parsed.query)
        query = (params.get("q") or [""])[0].strip()
        domains = [item.strip() for item in (params.get("domain") or [""])[0].split(",") if item.strip()]
        include_archive = (params.get("include_archive") or ["false"])[0].lower() == "true"
        try:
            limit = max(1, min(int((params.get("limit") or ["5"])[0]), 10))
        except ValueError:
            limit = 5
        self._search(query, domains, include_archive, limit)

    def do_POST(self):  # noqa: N802 - stdlib hook name
        if urlparse(self.path).path != "/search":
            self._send(404, {"ok": False, "error": "not_found"})
            return
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 64 * 1024)
            payload = json.loads(self.rfile.read(length) or b"{}")
            query = str(payload.get("query") or "").strip()
            domains = [str(item).strip() for item in payload.get("domains", []) if str(item).strip()]
            include_archive = bool(payload.get("include_archive", False))
            limit = max(1, min(int(payload.get("limit", 5)), 10))
        except (ValueError, TypeError, json.JSONDecodeError):
            self._send(400, {"ok": False, "error": "invalid_json"})
            return
        self._search(query, domains, include_archive, limit)

    def _search(self, query: str, domains: list[str], include_archive: bool, limit: int):
        if len(query) > 2000:
            self._send(400, {"ok": False, "error": "query_too_long"})
            return
        try:
            results = search_index(
                self.server.index_db,
                query,
                limit=limit,
                domains=domains,
                include_archive=include_archive,
                embedder=self.server.embedder,
            )
            self._send(200, {"ok": True, "read_only": True, "results": results})
        except Exception as exc:
            LOGGER.exception("recall search failed")
            self._send(500, {"ok": False, "error": "search_failed", "detail": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Ombre recall sidecar on loopback")
    parser.add_argument("--index-db", required=True)
    parser.add_argument("--host", default=os.environ.get("RECALL_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("RECALL_PORT", "8788")))
    parser.add_argument("--embedding-model", default=os.environ.get("RECALL_EMBEDDING_MODEL", ""))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    embedder = None
    # Load jieba's dictionary before opening the listener as well. Otherwise
    # the first request would pay a multi-second tokenizer cold-start cost.
    fts_text("sidecar warmup")
    if args.embedding_model:
        try:
            embedder = FastEmbedder(args.embedding_model)
            # ONNX may lazily initialize the session on the first embed call.
            # Warm it before opening the HTTP listener so healthz means the
            # first real request will not pay the cold-inference penalty.
            embedder.embed(["sidecar warmup"])
        except Exception as exc:
            LOGGER.warning("embedding backend unavailable; serve lexical-only: %s", exc)

    server = RecallServer((args.host, args.port), Handler, args.index_db, embedder)
    LOGGER.info("ombre recall sidecar listening on %s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
