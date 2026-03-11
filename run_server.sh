#!/usr/bin/env bash
set -euo pipefail

# Run the retrieval middleware FastAPI app from the project root.

PYTHONPATH=. uvicorn retrieval_middleware.main:app --reload --host 127.0.0.1 --port 8000

