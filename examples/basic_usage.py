"""Basic usage example: LightRAG with Google Cloud Spanner storage backend.

Prerequisites:
    1. Install packages:
        pip install lightrag-hku lightrag-spanner python-dotenv

    2. Copy .env.example to .env and fill in your settings.

    3. Authenticate with GCP:
        gcloud auth application-default login
"""

import asyncio

import lightrag_spanner
from lightrag import LightRAG, QueryParam

from _config import LLM_MODEL_NAME, SPANNER_ADDON_PARAMS, get_embedding_func
from lightrag.llm.gemini import gemini_model_complete

# Step 1: Register Spanner storage classes with LightRAG
lightrag_spanner.register()


async def main():
    # Step 2: Create a LightRAG instance with Spanner storage
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=gemini_model_complete,
        llm_model_name=LLM_MODEL_NAME,
        embedding_func=get_embedding_func(),
        kv_storage="SpannerKVStorage",
        vector_storage="SpannerVectorStorage",
        graph_storage="SpannerGraphStorage",
        doc_status_storage="SpannerDocStatusStorage",
        # Disable LLM caching — with remote storage like Spanner, cache lookups
        # add network round-trips on every LLM call with minimal hit rate benefit.
        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract=False,
        addon_params=SPANNER_ADDON_PARAMS,
    )

    # Step 3: Initialize storages (creates Spanner tables if needed)
    await rag.initialize_storages()

    # Step 4: Insert documents
    await rag.ainsert(
        "Artificial intelligence (AI) is the simulation of human intelligence "
        "processes by machines, especially computer systems. These processes "
        "include learning, reasoning, and self-correction."
    )

    # Step 5: Query using different modes
    # Hybrid mode: combines knowledge graph + vector retrieval
    result = await rag.aquery(
        "What is artificial intelligence?",
        param=QueryParam(mode="hybrid"),
    )
    print("=== Hybrid Mode ===")
    print(result)
    print()

    # Local mode: focuses on entity-level context
    result = await rag.aquery(
        "What processes does AI include?",
        param=QueryParam(mode="local"),
    )
    print("=== Local Mode ===")
    print(result)
    print()

    # Step 6: Clean up
    await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
