# lightrag-spanner

Google Cloud Spanner storage backend for [LightRAG](https://github.com/HKUDS/LightRAG).

This package provides four Spanner-backed storage classes as an external plugin — no modifications to LightRAG source code required.

| Storage | Class | Description |
|---|---|---|
| KV | `SpannerKVStorage` | Key-value storage with JSON serialization |
| Vector | `SpannerVectorStorage` | Vector storage with cosine similarity search |
| Graph | `SpannerGraphStorage` | Graph storage with Spanner Property Graph support |
| DocStatus | `SpannerDocStatusStorage` | Document processing status tracking |

## Installation

```bash
pip install lightrag-hku
pip install git+https://github.com/ksmin23/lightrag-spanner.git@v0.1.0
```

## Quick Start

```python
import asyncio
import lightrag_spanner
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

# Register Spanner storage classes with LightRAG
lightrag_spanner.register()

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
    await rag.ainsert("Your document text here")
    result = await rag.aquery("Your question", param=QueryParam(mode="hybrid"))
    print(result)
    await rag.finalize_storages()

asyncio.run(main())
```

## Configuration

Spanner connection settings can be provided via `addon_params` or environment variables. Environment variables are used as fallback when `addon_params` are not set.

| addon_params key | Environment Variable | Description |
|---|---|---|
| `spanner_project_id` | `GOOGLE_CLOUD_PROJECT` | GCP project ID |
| `spanner_instance_id` | `SPANNER_INSTANCE` | Spanner instance ID |
| `spanner_database_id` | `SPANNER_DATABASE` | Spanner database ID |
| `spanner_graph_name` | `SPANNER_GRAPH_NAME` | Property graph name (default: `knowledge_graph`) |

### Using Environment Variables

```bash
export GOOGLE_CLOUD_PROJECT=my-project
export SPANNER_INSTANCE=my-instance
export SPANNER_DATABASE=my-database
```

```python
lightrag_spanner.register()
rag = LightRAG(
    kv_storage="SpannerKVStorage",
    vector_storage="SpannerVectorStorage",
    graph_storage="SpannerGraphStorage",
    doc_status_storage="SpannerDocStatusStorage",
    ...
)
```

## Prerequisites

### GCP Authentication

```bash
gcloud auth application-default login
```

Or with a service account key:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

### Spanner Instance and Database

Create a Spanner instance and database beforehand. Tables are created automatically during initialization.

```bash
export SPANNER_INSTANCE=my-instance
export SPANNER_DATABASE=my-database
export GOOGLE_CLOUD_LOCATION="us-central1"

gcloud spanner instances create $SPANNER_INSTANCE \
    --config=regional-$GOOGLE_CLOUD_LOCATION \
    --description="LightRAG Instance" \
    --nodes=1 \
    --edition=ENTERPRISE

gcloud spanner databases create $SPANNER_DATABASE \
    --instance=$SPANNER_INSTANCE
```

## Project Structure

```
lightrag-spanner/
├── pyproject.toml
├── src/
│   └── lightrag_spanner/
│       ├── __init__.py      # register() and public exports
│       ├── client.py         # SpannerClientManager and helpers
│       └── storage.py        # All 4 storage class implementations
├── examples/
│   ├── README.md
│   ├── basic_usage.py
│   ├── env_var_config.py
│   ├── batch_insert_and_query.py
│   └── knowledge_graph_exploration.py
└── tests/
```

## Design Decisions

| Decision | Approach | Rationale |
|---|---|---|
| Sync vs Async | Synchronous Spanner SDK wrapped with `asyncio.to_thread` | Spanner Python SDK is synchronous; avoids blocking the event loop |
| Workspace Isolation | Column-based filtering (`WHERE workspace = @ws`) | Avoids DDL proliferation from per-workspace tables |
| Property Graph | GQL via Spanner Property Graph | Native graph traversal support; efficient BFS queries |
| Embedding Type | `ARRAY<FLOAT64>` | Spanner's native vector type with `COSINE_DISTANCE` support |
| Client Reuse | Singleton `SpannerClientManager` | Shares a single Spanner client across all storage classes |
| Fuzzy Search | `LIKE`-based pattern matching | Spanner lacks full-text search; LIKE is the pragmatic alternative |

## License

MIT
