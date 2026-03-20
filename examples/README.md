# lightrag-spanner Examples

A collection of examples demonstrating how to use Google Cloud Spanner as a storage backend for LightRAG.

## Prerequisites

### 1. Install Python Packages

```bash
pip install lightrag-hku lightrag-spanner
```

### 2. Authenticate with GCP

```bash
gcloud auth application-default login
```

Or, if using a service account key:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

### 3. Provision a Spanner Instance and Database

A Spanner instance and database must already exist in your GCP project.
Tables are created automatically by `lightrag-spanner` during initialization.

```bash
gcloud spanner instances create my-instance \
    --config=regional-us-central1 \
    --description="LightRAG Instance" \
    --nodes=1

gcloud spanner databases create my-database \
    --instance=my-instance
```

### 4. Configure Spanner Connection (choose one)

**Option A — Environment variables:**

```bash
export SPANNER_PROJECT=my-project
export SPANNER_INSTANCE=my-instance
export SPANNER_DATABASE=my-database
```

**Option B — addon_params (specified directly in code):**

```python
rag = LightRAG(
    addon_params={
        "spanner_project_id": "my-project",
        "spanner_instance_id": "my-instance",
        "spanner_database_id": "my-database",
    },
    ...
)
```

### 5. Set Your Gemini API Key

```bash
export GEMINI_API_KEY=...
```

## Examples

| File | Description |
|---|---|
| `basic_usage.py` | Basic usage. Insert a document and query with hybrid/local modes |
| `env_var_config.py` | Configure Spanner using environment variables only |
| `batch_insert_and_query.py` | Batch document insertion, compare 5 query modes, streaming responses |
| `knowledge_graph_exploration.py` | Direct graph storage access — entity browsing, BFS subgraph extraction |

## Running

```bash
cd examples
python basic_usage.py
```
