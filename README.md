Retrieval Middleware Proof-of-Concept
====================================

Running the app
---------------

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

