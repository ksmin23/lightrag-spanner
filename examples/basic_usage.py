"""Basic usage example: LightRAG with Google Cloud Spanner storage backend.

Prerequisites:
    1. Install packages:
        pip install lightrag-hku lightrag-spanner

    2. Configure Spanner access (choose one):
        a) Set environment variables:
            export SPANNER_PROJECT=my-project
            export SPANNER_INSTANCE=my-instance
            export SPANNER_DATABASE=my-database

        b) Or pass via addon_params (shown below)

    3. Set your LLM API key:
        export OPENAI_API_KEY=sk-...

    4. Authenticate with GCP:
        gcloud auth application-default login
"""

import asyncio

import lightrag_spanner
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

# Step 1: Register Spanner storage classes with LightRAG
lightrag_spanner.register()


async def main():
    # Step 2: Create a LightRAG instance with Spanner storage
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed,
        kv_storage="SpannerKVStorage",
        vector_storage="SpannerVectorStorage",
        graph_storage="SpannerGraphStorage",
        doc_status_storage="SpannerDocStatusStorage",
        addon_params={
            "spanner_project_id": "my-project",
            "spanner_instance_id": "my-instance",
            "spanner_database_id": "my-database",
        },
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
