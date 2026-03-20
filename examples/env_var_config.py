"""Environment variable configuration example.

Instead of passing Spanner settings via addon_params, you can use
environment variables. This is useful for containerized deployments
or when sharing config across multiple scripts.

Copy .env.example to .env and fill in your settings, or export them directly.
"""

import asyncio

import lightrag_spanner
from lightrag import LightRAG, QueryParam

from _config import LLM_MODEL_NAME, get_embedding_func
from lightrag.llm.gemini import gemini_model_complete

lightrag_spanner.register()


async def main():
    # No addon_params needed — Spanner config comes from env vars
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
    )

    await rag.initialize_storages()

    await rag.ainsert("The Python programming language was created by Guido van Rossum.")

    result = await rag.aquery(
        "Who created Python?",
        param=QueryParam(mode="mix"),
    )
    print(result)

    await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
