# Integration notes

`integration/recall-client.js` is intentionally isolated from the full companion bridge. The host application supplies:

- `content`: the current user message;
- `recentMessages`: a short recent conversation window;
- `sessionId` and `conversationId` for logging only;
- a logger implementing `info` and `warn`.

The host calls `runRecall(...)`, then passes `formatRecallContext(result.results)` into the model prompt. The client performs:

1. low-signal filtering;
2. optional model-led recall classification;
3. read-only sidecar retrieval;
4. optional model-led candidate reranking;
5. bounded context formatting with prompt-injection isolation.

The sidecar API must remain read-only. Its source code never receives a write operation from the client, and the actual index database should live outside the repository.
