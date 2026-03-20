"""Batch document insertion and multi-mode query example.

Demonstrates:
    - Inserting multiple documents at once
    - Querying with different modes (local, global, hybrid, naive, mix)
    - Using streaming responses
    - Tracking document processing status
"""

import argparse
import asyncio

import lightrag_spanner
from lightrag import LightRAG, QueryParam

from _config import LLM_MODEL_NAME, SPANNER_ADDON_PARAMS, get_embedding_func
from lightrag.llm.gemini import gemini_model_complete

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


async def main(cleanup: bool = False):
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=gemini_model_complete,
        llm_model_name=LLM_MODEL_NAME,
        embedding_func=get_embedding_func(),
        kv_storage="SpannerKVStorage",
        vector_storage="SpannerVectorStorage",
        graph_storage="SpannerGraphStorage",
        doc_status_storage="SpannerDocStatusStorage",
        # Disable LLM caching to avoid unnecessary Spanner round-trips
        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract=False,
        addon_params=SPANNER_ADDON_PARAMS,
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

    if cleanup:
        await rag.finalize_storages()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightRAG + Spanner batch insert and query example")
    parser.add_argument("--cleanup", action="store_true", help="Drop Spanner tables on exit")
    args = parser.parse_args()
    asyncio.run(main(cleanup=args.cleanup))
