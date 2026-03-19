"""Google Cloud Spanner storage backend for LightRAG."""

from lightrag_spanner.storage import (
    SpannerKVStorage,
    SpannerVectorStorage,
    SpannerGraphStorage,
    SpannerDocStatusStorage,
)

__all__ = [
    "SpannerKVStorage",
    "SpannerVectorStorage",
    "SpannerGraphStorage",
    "SpannerDocStatusStorage",
    "register",
]


def register() -> None:
    """Register Spanner storage classes with LightRAG's storage registry.

    Call this before creating a LightRAG instance with Spanner storage::

        import lightrag_spanner
        lightrag_spanner.register()

        rag = LightRAG(kv_storage="SpannerKVStorage", ...)
    """
    from lightrag.kg import (
        STORAGES,
        STORAGE_IMPLEMENTATIONS,
        STORAGE_ENV_REQUIREMENTS,
    )

    STORAGES.update(
        {
            "SpannerKVStorage": "lightrag_spanner.storage",
            "SpannerVectorStorage": "lightrag_spanner.storage",
            "SpannerGraphStorage": "lightrag_spanner.storage",
            "SpannerDocStatusStorage": "lightrag_spanner.storage",
        }
    )

    STORAGE_IMPLEMENTATIONS["KV_STORAGE"]["implementations"].append(
        "SpannerKVStorage"
    )
    STORAGE_IMPLEMENTATIONS["VECTOR_STORAGE"]["implementations"].append(
        "SpannerVectorStorage"
    )
    STORAGE_IMPLEMENTATIONS["GRAPH_STORAGE"]["implementations"].append(
        "SpannerGraphStorage"
    )
    STORAGE_IMPLEMENTATIONS["DOC_STATUS_STORAGE"]["implementations"].append(
        "SpannerDocStatusStorage"
    )

    env_reqs = ["SPANNER_INSTANCE", "SPANNER_DATABASE"]
    STORAGE_ENV_REQUIREMENTS.update(
        {
            "SpannerKVStorage": env_reqs,
            "SpannerVectorStorage": env_reqs,
            "SpannerGraphStorage": env_reqs,
            "SpannerDocStatusStorage": env_reqs,
        }
    )
