# lightrag-spanner Examples

A collection of examples demonstrating how to use Google Cloud Spanner as a storage backend for LightRAG.

## Prerequisites

### 1. Install Python Packages

```bash
pip install lightrag-hku
pip install git+https://github.com/ksmin23/lightrag-spanner.git@v0.1.0
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

### 4. Configure Spanner Connection (choose one)

**Option A — Environment variables:**

```bash
export GOOGLE_CLOUD_PROJECT=my-project
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

### 5. Configure LLM Authentication

**Option A — Gemini via Vertex AI** (recommended when running on GCP):

```bash
export GOOGLE_GENAI_USE_VERTEXAI=true
```

Uses Application Default Credentials (ADC). No API key needed.

**Option B — Gemini via AI Studio**:

```bash
export GEMINI_API_KEY=your-gemini-api-key
```

**Option C — OpenAI**:

```bash
export OPENAI_API_KEY=your-openai-api-key
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
