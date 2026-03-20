"""Shared configuration loader for examples.

Reads model and Spanner settings from .env (via dotenv) or environment variables,
and builds the LLM / embedding function objects used by all examples.

Environment variables (all optional — defaults shown):
    GOOGLE_CLOUD_PROJECT     = (required for Spanner)
    SPANNER_INSTANCE         = (required for Spanner)
    SPANNER_DATABASE         = (required for Spanner)
    LLM_MODEL_NAME           = gemini-2.5-flash
    EMBEDDING_MODEL_NAME     = gemini-embedding-001
    EMBEDDING_DIM            = 1536
    EMBEDDING_MAX_TOKEN_SIZE = 2048
"""

import os
from dotenv import find_dotenv, load_dotenv

from lightrag.llm.gemini import gemini_embed, gemini_model_complete
from lightrag.utils import EmbeddingFunc

# Load .env from the closest parent directory (if present)
load_dotenv(find_dotenv(usecwd=True))

# --- Spanner settings ---
SPANNER_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
SPANNER_INSTANCE = os.getenv("SPANNER_INSTANCE", "")
SPANNER_DATABASE = os.getenv("SPANNER_DATABASE", "")

SPANNER_ADDON_PARAMS = {
    "spanner_project_id": SPANNER_PROJECT,
    "spanner_instance_id": SPANNER_INSTANCE,
    "spanner_database_id": SPANNER_DATABASE,
}

# --- LLM settings ---
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")

# --- Embedding settings ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "gemini-embedding-001")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
EMBEDDING_MAX_TOKEN_SIZE = int(os.getenv("EMBEDDING_MAX_TOKEN_SIZE", "2048"))


def get_embedding_func() -> EmbeddingFunc:
    """Build an EmbeddingFunc with settings from environment variables."""
    return EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
        model_name=EMBEDDING_MODEL_NAME,
        func=gemini_embed.func,
        send_dimensions=True,
    )
