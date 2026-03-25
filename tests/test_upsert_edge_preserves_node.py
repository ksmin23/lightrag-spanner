"""Test that upsert_edge does NOT overwrite existing node data.

Verifies the fix for the bug where upsert_edge called upsert_node with
empty data, wiping out entity_type and description of existing nodes.

Prerequisites:
    1. Install packages:
        pip install lightrag-hku lightrag-spanner python-dotenv

    2. Copy examples/.env.example to examples/.env and fill in your settings.

    3. Authenticate with GCP:
        gcloud auth application-default login

Usage:
    python -m tests.test_upsert_edge_preserves_node
"""

import os
import asyncio
import logging

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

import lightrag_spanner
from lightrag import LightRAG
from lightrag.llm.gemini import gemini_embed, gemini_model_complete
from lightrag.utils import EmbeddingFunc

logger = logging.getLogger(__name__)

lightrag_spanner.register()

WORKSPACE = os.getenv("WORKSPACE", "lightrag")
TEST_WORKSPACE = f"{WORKSPACE}_test"


def get_embedding_func() -> EmbeddingFunc:
    return EmbeddingFunc(
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "1536")),
        max_token_size=int(os.getenv("EMBEDDING_MAX_TOKEN_SIZE", "2048")),
        model_name=os.getenv("EMBEDDING_MODEL_NAME", "gemini-embedding-001"),
        func=gemini_embed.func,
        send_dimensions=True,
    )


async def main():
    rag = LightRAG(
        working_dir="./rag_storage",
        workspace=TEST_WORKSPACE,
        llm_model_func=gemini_model_complete,
        llm_model_name=os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash"),
        embedding_func=get_embedding_func(),
        kv_storage="SpannerKVStorage",
        vector_storage="SpannerVectorStorage",
        graph_storage="SpannerGraphStorage",
        doc_status_storage="SpannerDocStatusStorage",
        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract=False,
        addon_params={
            "spanner_project_id": os.getenv("GOOGLE_CLOUD_PROJECT", ""),
            "spanner_instance_id": os.getenv("SPANNER_INSTANCE", ""),
            "spanner_database_id": os.getenv("SPANNER_DATABASE", ""),
            "spanner_graph_name": os.getenv("SPANNER_GRAPH_NAME", "lightrag_test_knowledge_graph"),
        },
    )
    await rag.initialize_storages()

    graph = rag.chunk_entity_relation_graph

    # Clean up any previous test data
    await graph.drop()

    passed = 0
    failed = 0

    # --- Test 1: upsert_node stores entity_type and description ---
    print("Test 1: upsert_node stores entity_type and description")
    await graph.upsert_node("Alice", {
        "entity_type": "person",
        "description": "A software engineer",
        "source_id": "chunk-1",
    })
    node = await graph.get_node("Alice")
    assert node is not None, "Node 'Alice' should exist"
    if node["entity_type"] == "person" and node["description"] == "A software engineer":
        print("  PASSED")
        passed += 1
    else:
        print(f"  FAILED: expected (person, A software engineer), got ({node['entity_type']}, {node['description']})")
        failed += 1

    # --- Test 2: upsert_edge preserves existing node data ---
    print("Test 2: upsert_edge preserves existing node data")
    await graph.upsert_edge("Alice", "Bob", {
        "weight": "1.0",
        "description": "Alice knows Bob",
        "keywords": "friendship",
        "source_id": "chunk-1",
    })
    node = await graph.get_node("Alice")
    assert node is not None, "Node 'Alice' should still exist"
    if node["entity_type"] == "person" and node["description"] == "A software engineer":
        print("  PASSED")
        passed += 1
    else:
        print(f"  FAILED: expected (person, A software engineer), got ({node['entity_type']}, {node['description']})")
        failed += 1

    # --- Test 3: upsert_edge creates missing node with empty fields ---
    print("Test 3: upsert_edge creates missing node with empty fields")
    node_bob = await graph.get_node("Bob")
    assert node_bob is not None, "Node 'Bob' should be created by upsert_edge"
    if node_bob["entity_type"] == "" and node_bob["description"] == "":
        print("  PASSED")
        passed += 1
    else:
        print(f"  FAILED: expected empty fields, got ({node_bob['entity_type']}, {node_bob['description']})")
        failed += 1

    # --- Test 4: upsert_node after upsert_edge updates the placeholder node ---
    print("Test 4: upsert_node updates placeholder node created by upsert_edge")
    await graph.upsert_node("Bob", {
        "entity_type": "person",
        "description": "A data scientist",
        "source_id": "chunk-2",
    })
    node_bob = await graph.get_node("Bob")
    if node_bob["entity_type"] == "person" and node_bob["description"] == "A data scientist":
        print("  PASSED")
        passed += 1
    else:
        print(f"  FAILED: expected (person, A data scientist), got ({node_bob['entity_type']}, {node_bob['description']})")
        failed += 1

    # --- Test 5: edge data is correct ---
    print("Test 5: edge data is stored correctly")
    edge = await graph.get_edge("Alice", "Bob")
    assert edge is not None, "Edge Alice->Bob should exist"
    if edge["description"] == "Alice knows Bob" and edge["keywords"] == "friendship":
        print("  PASSED")
        passed += 1
    else:
        print(f"  FAILED: got ({edge['description']}, {edge['keywords']})")
        failed += 1

    # Clean up: drop test tables from Spanner
    await _drop_test_tables(rag)
    await rag.finalize_storages()

    print(f"\nResults: {passed} passed, {failed} failed")
    if failed > 0:
        raise SystemExit(1)


async def _drop_test_tables(rag):
    """Drop all Spanner tables and property graph created during the test."""
    graph = rag.chunk_entity_relation_graph
    db = graph._manager.database

    # Collect all table names from every storage instance
    table_names = set()
    storages = [
        rag.full_docs, rag.text_chunks,
        rag.full_entities, rag.full_relations,
        rag.entity_chunks, rag.relation_chunks,
        rag.entities_vdb, rag.relationships_vdb, rag.chunks_vdb,
        rag.llm_response_cache, rag.doc_status,
    ]
    for s in storages:
        if s and hasattr(s, "_table_name"):
            table_names.add(s._table_name)

    # Property graph and graph tables must be dropped in order
    ddl_statements = [
        f"DROP PROPERTY GRAPH IF EXISTS {graph._graph_name}",
        f"DROP TABLE IF EXISTS {graph._edges_table}",
        f"DROP TABLE IF EXISTS {graph._nodes_table}",
    ]
    # Then drop all kv/vector/doc_status tables
    for t in sorted(table_names):
        ddl_statements.append(f"DROP TABLE IF EXISTS {t}")

    for ddl in ddl_statements:
        try:
            op = db.update_ddl([ddl])
            await asyncio.to_thread(op.result)
        except Exception:
            logger.error("DDL failed: %s", ddl, exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
