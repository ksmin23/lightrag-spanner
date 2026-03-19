"""Environment variable configuration example.

Instead of passing Spanner settings via addon_params, you can use
environment variables. This is useful for containerized deployments
or when sharing config across multiple scripts.

Set the following before running:
    export SPANNER_PROJECT=my-project
    export SPANNER_INSTANCE=my-instance
    export SPANNER_DATABASE=my-database
    export OPENAI_API_KEY=sk-...
"""

import asyncio

import lightrag_spanner
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

lightrag_spanner.register()


async def main():
    # No addon_params needed — Spanner config comes from env vars
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed,
        kv_storage="SpannerKVStorage",
        vector_storage="SpannerVectorStorage",
        graph_storage="SpannerGraphStorage",
        doc_status_storage="SpannerDocStatusStorage",
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
