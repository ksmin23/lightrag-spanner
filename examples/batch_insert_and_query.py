"""Batch document insertion and multi-mode query example.

Demonstrates:
    - Inserting multiple documents at once
    - Querying with different modes (local, global, hybrid, naive, mix)
    - Using streaming responses
    - Tracking document processing status
"""

import asyncio

import lightrag_spanner
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

lightrag_spanner.register()

DOCUMENTS = [
    (
        "Albert Einstein developed the theory of relativity, one of the two "
        "pillars of modern physics. His work is also known for its influence "
        "on the philosophy of science."
    ),
    (
        "Isaac Newton formulated the laws of motion and universal gravitation "
        "that dominated scientists' view of the physical universe for nearly "
        "three centuries."
    ),
    (
        "Marie Curie was a Polish-French physicist and chemist who conducted "
        "pioneering research on radioactivity. She was the first woman to win "
        "a Nobel Prize."
    ),
]


async def main():
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

    await rag.initialize_storages()

    # --- Batch insert ---
    print("Inserting documents...")
    await rag.ainsert(DOCUMENTS)
    print(f"Inserted {len(DOCUMENTS)} documents.\n")

    # --- Query with different modes ---
    question = "What are the major contributions of these scientists?"

    for mode in ("local", "global", "hybrid", "naive", "mix"):
        result = await rag.aquery(question, param=QueryParam(mode=mode))
        print(f"=== {mode.upper()} mode ===")
        print(result)
        print()

    # --- Streaming response ---
    print("=== Streaming ===")
    response = await rag.aquery(
        "Tell me about Marie Curie.",
        param=QueryParam(mode="hybrid", stream=True),
    )
    async for chunk in response:
        print(chunk, end="", flush=True)
    print("\n")

    await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
