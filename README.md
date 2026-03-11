Retrieval Middleware Proof-of-Concept
====================================

## What this does

A **lightweight middleware layer** that sits between a user query and a vector database. It adds:

- **Semantic cache** — Reuses results for similar queries (cosine similarity above a threshold) so repeated or paraphrased questions skip the vector DB and reranker.

- **Re-ranking** — Takes the top-k candidates from the vector DB and re-scores them with a cross-encoder for better relevance.

- **Latency visibility** — Every response includes a breakdown of time spent in embedding, cache lookup, vector DB call, and reranking.

**Tech:** FastAPI, sentence-transformers (embeddings + cross-encoder), in-memory cache, mock vector DB with simulated latency. No real database required to run.

**Flow:** Query → embed → check cache → on miss: vector DB (top 50) → rerank to top_n → cache result → return JSON with results and timing.

---

## Running the app

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the server (from project root, so the package imports work):

   ```bash
   cd /path/to/sub\ latency\ layer
   PYTHONPATH=. uvicorn retrieval_middleware.main:app --reload --host 127.0.0.1 --port 8000
   ```

   The server binds immediately. The first `/query` may take 30–60s while embedding and reranker models load.

3. Explore the API via Swagger UI:

   Open `http://127.0.0.1:8000/docs` or `http://localhost:8000/docs` in your browser.

4. Run tests (from project root):

   ```bash
   PYTHONPATH=. pytest retrieval_middleware/tests -v
   ```

5. Sample curl — cold query:

   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What is machine learning?", "top_n": 5}'
   ```

6. Sample curl — warm query:

   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What is machine learning?", "top_n": 5}'
   ```

7. Cache stats:

   ```bash
   curl http://localhost:8000/stats
   ```

8. Clear the cache:

   ```bash
   curl -X DELETE http://localhost:8000/cache
   ```

