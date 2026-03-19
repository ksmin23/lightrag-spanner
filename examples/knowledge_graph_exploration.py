"""Knowledge graph exploration example.

Demonstrates direct access to the Spanner graph storage for:
    - Browsing nodes and edges
    - Searching entity labels
    - Extracting a subgraph via BFS
"""

import asyncio
import json

import lightrag_spanner
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

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

    await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
