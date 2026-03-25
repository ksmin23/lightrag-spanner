"""Knowledge graph exploration example.

Demonstrates direct access to the Spanner graph storage for:
    - Browsing nodes and edges
    - Searching entity labels
    - Extracting a subgraph via BFS
"""

import argparse
import asyncio

import lightrag_spanner
from lightrag import LightRAG, QueryParam

from _config import LLM_MODEL_NAME, SPANNER_ADDON_PARAMS, WORKSPACE, get_embedding_func
from lightrag.llm.gemini import gemini_model_complete

lightrag_spanner.register()


async def main(cleanup: bool = False):
    rag = LightRAG(
        working_dir="./rag_storage",
        workspace=WORKSPACE,
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

    # Insert some data first
    await rag.ainsert(
        "Google Cloud Spanner is a fully managed relational database service "
        "that runs on Google Cloud. It combines the benefits of relational "
        "database structure with non-relational horizontal scale."
    )

    # Access the graph storage directly
    graph = rag.chunk_entity_relation_graph

    # List all entity labels
    labels = await graph.get_all_labels()
    print(f"Entity labels ({len(labels)}):")
    for label in labels[:20]:
        print(f"  - {label}")
    print()

    # Get popular labels (sorted by degree)
    popular = await graph.get_popular_labels(limit=10)
    print(f"Most connected entities:")
    for label in popular:
        degree = await graph.node_degree(label)
        print(f"  - {label} (degree: {degree})")
    print()

    # Search labels
    if labels:
        search_term = labels[0][:3]  # use first few chars of first label
        matches = await graph.search_labels(search_term, limit=5)
        print(f"Search results for '{search_term}': {matches}")
        print()

    # Extract a subgraph using BFS
    if labels:
        kg = await graph.get_knowledge_graph(
            node_label=labels[0],
            max_depth=2,
            max_nodes=50,
        )
        print(f"Subgraph for '{labels[0]}':")
        print(f"  Nodes: {len(kg.nodes)}")
        print(f"  Edges: {len(kg.edges)}")
        print(f"  Truncated: {kg.is_truncated}")

        for node in kg.nodes[:5]:
            print(f"  Node: {node.id} ({node.labels})")
        for edge in kg.edges[:5]:
            print(f"  Edge: {edge.source} -> {edge.target} [{edge.type}]")

    if cleanup:
        await rag.finalize_storages()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightRAG + Spanner knowledge graph exploration example")
    parser.add_argument("--cleanup", action="store_true", help="Drop Spanner tables on exit")
    args = parser.parse_args()
    asyncio.run(main(cleanup=args.cleanup))
