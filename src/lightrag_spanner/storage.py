"""Spanner storage implementations for LightRAG.

This module provides four storage classes:
- SpannerKVStorage       (BaseKVStorage)
- SpannerVectorStorage   (BaseVectorStorage)
- SpannerGraphStorage    (BaseGraphStorage)
- SpannerDocStatusStorage (DocStatusStorage)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, final

import numpy as np
from google.cloud import spanner

from lightrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
)
from lightrag.types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode
from lightrag.utils import compute_mdhash_id, logger as lightrag_logger

from lightrag_spanner.client import (
    SpannerClientManager,
    _ensure_table,
    get_spanner_config,
)

logger = logging.getLogger("lightrag.spanner")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_table_name(workspace: str, suffix: str) -> str:
    """Build a table name from workspace + suffix, e.g. ``myws_kv``."""
    return f"{workspace}_{suffix}" if workspace else suffix


def _run_sync(fn, *args, **kwargs):
    """Run a synchronous Spanner SDK call in a thread to avoid blocking the event loop."""
    return asyncio.to_thread(fn, *args, **kwargs)


# ===================================================================
# KV Storage
# ===================================================================

@final
@dataclass
class SpannerKVStorage(BaseKVStorage):
    _manager: SpannerClientManager | None = field(default=None, repr=False)

    def __post_init__(self):
        self._max_batch_size = self.global_config.get("embedding_batch_num", 20)
        cfg = get_spanner_config(self.global_config)
        self._spanner_cfg = cfg
        self._table_name = _make_table_name(self.workspace, f"{self.namespace}_kv")

    async def initialize(self):
        if self._manager is None:
            cfg = self._spanner_cfg
            self._manager = await SpannerClientManager.get_instance(
                cfg["project_id"], cfg["instance_id"], cfg["database_id"]
            )
        db = self._manager.database
        ddl = [
            f"""CREATE TABLE {self._table_name} (
                id STRING(MAX) NOT NULL,
                workspace STRING(MAX) NOT NULL,
                data STRING(MAX)
            ) PRIMARY KEY (id, workspace)"""
        ]
        await _run_sync(_ensure_table, db, self._table_name, ddl)
        logger.info("SpannerKVStorage initialized: %s", self._table_name)

    async def finalize(self):
        if self._manager is not None:
            await SpannerClientManager.release_instance(self._manager)
            self._manager = None

    async def index_done_callback(self) -> None:
        # Spanner provides immediate consistency — no-op.
        pass

    # --- reads ---

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        def _query(txn):
            result = txn.execute_sql(
                f"SELECT data FROM {self._table_name} WHERE id = @id AND workspace = @ws",
                params={"id": id, "ws": self.workspace},
                param_types={"id": spanner.param_types.STRING, "ws": spanner.param_types.STRING},
            )
            rows = list(result)
            return rows[0][0] if rows else None

        raw = await _run_sync(self._manager.database.run_in_transaction, _query)
        if raw is None:
            return None
        return json.loads(raw)

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []

        def _query(txn):
            result = txn.execute_sql(
                f"SELECT id, data FROM {self._table_name} "
                "WHERE id IN UNNEST(@ids) AND workspace = @ws",
                params={"ids": ids, "ws": self.workspace},
                param_types={
                    "ids": spanner.param_types.Array(spanner.param_types.STRING),
                    "ws": spanner.param_types.STRING,
                },
            )
            return {row[0]: row[1] for row in result}

        rows_map = await _run_sync(self._manager.database.run_in_transaction, _query)
        results = []
        for doc_id in ids:
            raw = rows_map.get(doc_id)
            results.append(json.loads(raw) if raw else None)
        return results

    async def filter_keys(self, keys: set[str]) -> set[str]:
        if not keys:
            return set()
        keys_list = list(keys)

        def _query(txn):
            result = txn.execute_sql(
                f"SELECT id FROM {self._table_name} "
                "WHERE id IN UNNEST(@ids) AND workspace = @ws",
                params={"ids": keys_list, "ws": self.workspace},
                param_types={
                    "ids": spanner.param_types.Array(spanner.param_types.STRING),
                    "ws": spanner.param_types.STRING,
                },
            )
            return {row[0] for row in result}

        existing = await _run_sync(self._manager.database.run_in_transaction, _query)
        return keys - existing

    # --- writes ---

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return

        def _write(txn):
            rows = []
            for k, v in data.items():
                rows.append((k, self.workspace, json.dumps(v, ensure_ascii=False)))
            txn.insert_or_update(
                self._table_name,
                columns=["id", "workspace", "data"],
                values=rows,
            )

        await _run_sync(self._manager.database.run_in_transaction, _write)

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return

        def _delete(txn):
            keyset = spanner.KeySet(keys=[[doc_id, self.workspace] for doc_id in ids])
            txn.delete(self._table_name, keyset)

        await _run_sync(self._manager.database.run_in_transaction, _delete)

    async def is_empty(self) -> bool:
        def _query(txn):
            result = txn.execute_sql(
                f"SELECT 1 FROM {self._table_name} WHERE workspace = @ws LIMIT 1",
                params={"ws": self.workspace},
                param_types={"ws": spanner.param_types.STRING},
            )
            return len(list(result)) == 0

        return await _run_sync(self._manager.database.run_in_transaction, _query)

    async def drop(self) -> dict[str, str]:
        try:

            def _drop(txn):
                txn.execute_update(
                    f"DELETE FROM {self._table_name} WHERE workspace = @ws",
                    params={"ws": self.workspace},
                    param_types={"ws": spanner.param_types.STRING},
                )

            await _run_sync(self._manager.database.run_in_transaction, _drop)
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error("SpannerKVStorage.drop failed: %s", e)
            return {"status": "error", "message": str(e)}


# ===================================================================
# Vector Storage
# ===================================================================

@final
@dataclass
class SpannerVectorStorage(BaseVectorStorage):
    _manager: SpannerClientManager | None = field(default=None, repr=False)

    def __post_init__(self):
        self._validate_embedding_func()
        cfg = get_spanner_config(self.global_config)
        self._spanner_cfg = cfg
        self._max_batch_size = self.global_config.get("embedding_batch_num", 20)

        # Build table name with optional embedding model suffix
        suffix = self._generate_collection_suffix()
        ns = self.namespace
        if suffix:
            ns = f"{ns}_{suffix}"
        self._table_name = _make_table_name(self.workspace, f"vdb_{ns}")

        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        threshold = kwargs.get("cosine_better_than_threshold")
        if threshold is not None:
            self.cosine_better_than_threshold = threshold

        # Exclude columns already defined as fixed schema columns
        _fixed_cols = {"id", "workspace", "embedding", "content", "created_at"}
        self._extra_meta = sorted(self.meta_fields - _fixed_cols)

    async def initialize(self):
        if self._manager is None:
            cfg = self._spanner_cfg
            self._manager = await SpannerClientManager.get_instance(
                cfg["project_id"], cfg["instance_id"], cfg["database_id"]
            )

        # Build columns: id, workspace, embedding, content, + extra meta fields
        meta_cols = "".join(
            f",\n    {f} STRING(MAX)" for f in self._extra_meta
        )
        ddl = [
            f"""CREATE TABLE {self._table_name} (
    id STRING(MAX) NOT NULL,
    workspace STRING(MAX) NOT NULL,
    embedding ARRAY<FLOAT64>,
    content STRING(MAX),
    created_at INT64{meta_cols}
) PRIMARY KEY (id, workspace)"""
        ]
        db = self._manager.database
        await _run_sync(_ensure_table, db, self._table_name, ddl)
        logger.info("SpannerVectorStorage initialized: %s", self._table_name)

    async def finalize(self):
        if self._manager is not None:
            await SpannerClientManager.release_instance(self._manager)
            self._manager = None

    async def index_done_callback(self) -> None:
        pass

    # --- upsert ---

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return

        current_time = int(time.time())
        list_data = []
        for k, v in data.items():
            entry: dict[str, Any] = {"id": k, "workspace": self.workspace, "created_at": current_time}
            for mf in self._extra_meta:
                entry[mf] = str(v.get(mf, ""))
            list_data.append(entry)

        # Compute embeddings in batches
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)
        embeddings = np.concatenate(embeddings_list)

        for i, entry in enumerate(list_data):
            entry["embedding"] = embeddings[i].tolist()
            entry["content"] = contents[i]

        columns = ["id", "workspace", "embedding", "content", "created_at"] + self._extra_meta

        def _write(txn):
            rows = []
            for entry in list_data:
                rows.append(tuple(entry.get(c) for c in columns))
            txn.insert_or_update(self._table_name, columns=columns, values=rows)

        await _run_sync(self._manager.database.run_in_transaction, _write)

    # --- query ---

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        if query_embedding is not None:
            if hasattr(query_embedding, "tolist"):
                qvec = query_embedding.tolist()
            else:
                qvec = list(query_embedding)
        else:
            emb = await self.embedding_func([query], _priority=5)
            qvec = emb[0].tolist()

        meta_cols = ", ".join(self._extra_meta) if self._extra_meta else ""
        select_cols = "id, content, created_at"
        if meta_cols:
            select_cols += f", {meta_cols}"

        # COSINE_DISTANCE returns 0 (identical) to 2 (opposite).
        # Convert similarity threshold: max_distance = 1 - similarity_threshold
        max_distance = 1.0 - self.cosine_better_than_threshold

        def _query(txn):
            result = txn.execute_sql(
                f"SELECT {select_cols}, "
                f"COSINE_DISTANCE(embedding, @qvec) AS dist "
                f"FROM {self._table_name} "
                "WHERE workspace = @ws "
                "AND embedding IS NOT NULL "
                "ORDER BY dist ASC "
                "LIMIT @top_k",
                params={
                    "ws": self.workspace,
                    "qvec": qvec,
                    "top_k": top_k,
                },
                param_types={
                    "ws": spanner.param_types.STRING,
                    "qvec": spanner.param_types.Array(spanner.param_types.FLOAT64),
                    "top_k": spanner.param_types.INT64,
                },
            )
            return list(result)

        rows = await _run_sync(self._manager.database.run_in_transaction, _query)
        if not rows:
            return []

        num_meta = len(self._extra_meta)
        scored = []
        for row in rows:
            row_id = row[0]
            content = row[1]
            created_at = row[2]
            meta_values = row[3 : 3 + num_meta] if num_meta else []
            cosine_dist = row[3 + num_meta]

            # Filter by similarity threshold (distance <= max_distance)
            if cosine_dist > max_distance:
                continue

            doc: dict[str, Any] = {
                "id": row_id,
                "content": content,
                "created_at": created_at,
                "distance": 1.0 - cosine_dist,  # Convert back to similarity
            }
            for j, mf in enumerate(self._extra_meta):
                doc[mf] = meta_values[j] if j < len(meta_values) else ""
            scored.append(doc)

        return scored

    # --- get / delete ---

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        meta_cols = ", ".join(self._extra_meta) if self._extra_meta else ""
        select_cols = "id, content, created_at"
        if meta_cols:
            select_cols += f", {meta_cols}"

        def _query(txn):
            result = txn.execute_sql(
                f"SELECT {select_cols} FROM {self._table_name} "
                "WHERE id = @id AND workspace = @ws",
                params={"id": id, "ws": self.workspace},
                param_types={"id": spanner.param_types.STRING, "ws": spanner.param_types.STRING},
            )
            return list(result)

        rows = await _run_sync(self._manager.database.run_in_transaction, _query)
        if not rows:
            return None
        row = rows[0]
        doc: dict[str, Any] = {"id": row[0], "content": row[1], "created_at": row[2]}
        sorted_meta = self._extra_meta
        for j, mf in enumerate(sorted_meta):
            doc[mf] = row[3 + j] if 3 + j < len(row) else ""
        return doc

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        meta_cols = ", ".join(self._extra_meta) if self._extra_meta else ""
        select_cols = "id, content, created_at"
        if meta_cols:
            select_cols += f", {meta_cols}"

        def _query(txn):
            result = txn.execute_sql(
                f"SELECT {select_cols} FROM {self._table_name} "
                "WHERE id IN UNNEST(@ids) AND workspace = @ws",
                params={"ids": ids, "ws": self.workspace},
                param_types={
                    "ids": spanner.param_types.Array(spanner.param_types.STRING),
                    "ws": spanner.param_types.STRING,
                },
            )
            return {r[0]: r for r in result}

        rows_map = await _run_sync(self._manager.database.run_in_transaction, _query)
        results = []
        sorted_meta = self._extra_meta
        for doc_id in ids:
            row = rows_map.get(doc_id)
            if row is None:
                results.append(None)
                continue
            doc: dict[str, Any] = {"id": row[0], "content": row[1], "created_at": row[2]}
            for j, mf in enumerate(sorted_meta):
                doc[mf] = row[3 + j] if 3 + j < len(row) else ""
            results.append(doc)
        return results

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        if not ids:
            return {}

        def _query(txn):
            result = txn.execute_sql(
                f"SELECT id, embedding FROM {self._table_name} "
                "WHERE id IN UNNEST(@ids) AND workspace = @ws",
                params={"ids": ids, "ws": self.workspace},
                param_types={
                    "ids": spanner.param_types.Array(spanner.param_types.STRING),
                    "ws": spanner.param_types.STRING,
                },
            )
            return {r[0]: list(r[1]) for r in result if r[1]}

        return await _run_sync(self._manager.database.run_in_transaction, _query)

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return

        def _delete(txn):
            keyset = spanner.KeySet(keys=[[doc_id, self.workspace] for doc_id in ids])
            txn.delete(self._table_name, keyset)

        await _run_sync(self._manager.database.run_in_transaction, _delete)

    async def delete_entity(self, entity_name: str) -> None:
        entity_id = compute_mdhash_id(entity_name, prefix="ent-")
        await self.delete([entity_id])

    async def delete_entity_relation(self, entity_name: str) -> None:
        def _delete(txn):
            txn.execute_update(
                f"DELETE FROM {self._table_name} "
                "WHERE workspace = @ws AND (src_id = @name OR tgt_id = @name)",
                params={"ws": self.workspace, "name": entity_name},
                param_types={
                    "ws": spanner.param_types.STRING,
                    "name": spanner.param_types.STRING,
                },
            )

        await _run_sync(self._manager.database.run_in_transaction, _delete)

    async def drop(self) -> dict[str, str]:
        try:

            def _drop(txn):
                txn.execute_update(
                    f"DELETE FROM {self._table_name} WHERE workspace = @ws",
                    params={"ws": self.workspace},
                    param_types={"ws": spanner.param_types.STRING},
                )

            await _run_sync(self._manager.database.run_in_transaction, _drop)
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error("SpannerVectorStorage.drop failed: %s", e)
            return {"status": "error", "message": str(e)}


# ===================================================================
# Graph Storage
# ===================================================================

@final
@dataclass
class SpannerGraphStorage(BaseGraphStorage):
    _manager: SpannerClientManager | None = field(default=None, repr=False)

    def __post_init__(self):
        cfg = get_spanner_config(self.global_config)
        self._spanner_cfg = cfg
        self._graph_name = cfg.get("graph_name", "lightrag_knowledge_graph")
        self._nodes_table = _make_table_name(self.workspace, "nodes")
        self._edges_table = _make_table_name(self.workspace, "edges")

    async def initialize(self):
        if self._manager is None:
            cfg = self._spanner_cfg
            self._manager = await SpannerClientManager.get_instance(
                cfg["project_id"], cfg["instance_id"], cfg["database_id"]
            )

        db = self._manager.database

        nodes_ddl = [
            f"""CREATE TABLE {self._nodes_table} (
    id STRING(MAX) NOT NULL,
    workspace STRING(MAX) NOT NULL,
    entity_type STRING(MAX),
    description STRING(MAX),
    source_id STRING(MAX)
) PRIMARY KEY (id, workspace)"""
        ]
        edges_ddl = [
            f"""CREATE TABLE {self._edges_table} (
    id STRING(MAX) NOT NULL,
    target_id STRING(MAX) NOT NULL,
    workspace STRING(MAX) NOT NULL,
    weight FLOAT64,
    description STRING(MAX),
    keywords STRING(MAX),
    source_id STRING(MAX)
) PRIMARY KEY (id, target_id, workspace)"""
        ]
        await _run_sync(_ensure_table, db, self._nodes_table, nodes_ddl)
        await _run_sync(_ensure_table, db, self._edges_table, edges_ddl)

        # Create property graph (best-effort — may already exist)
        graph_ddl = [
            f"""CREATE OR REPLACE PROPERTY GRAPH {self._graph_name}
    NODE TABLES (
        {self._nodes_table}
            KEY(id, workspace)
            LABEL Entity
                PROPERTIES(id, entity_type, description, source_id)
    )
    EDGE TABLES (
        {self._edges_table}
            KEY(id, target_id, workspace)
            SOURCE KEY(id, workspace) REFERENCES {self._nodes_table}(id, workspace)
            DESTINATION KEY(target_id, workspace) REFERENCES {self._nodes_table}(id, workspace)
            LABEL Relationship
                PROPERTIES(weight, description, keywords, source_id)
    )"""
        ]
        try:
            op = db.update_ddl(graph_ddl)
            await _run_sync(op.result)
        except Exception as e:
            logger.error("Property graph DDL failed: %s", e, exc_info=True)
            raise

        logger.info(
            "SpannerGraphStorage initialized: nodes=%s, edges=%s, graph=%s",
            self._nodes_table,
            self._edges_table,
            self._graph_name,
        )

    async def finalize(self):
        if self._manager is not None:
            await SpannerClientManager.release_instance(self._manager)
            self._manager = None

    async def index_done_callback(self) -> None:
        pass

    # --- node CRUD ---

    async def has_node(self, node_id: str) -> bool:
        def _q(txn):
            rows = list(
                txn.execute_sql(
                    f"SELECT 1 FROM {self._nodes_table} WHERE id = @id AND workspace = @ws LIMIT 1",
                    params={"id": node_id, "ws": self.workspace},
                    param_types={"id": spanner.param_types.STRING, "ws": spanner.param_types.STRING},
                )
            )
            return len(rows) > 0

        return await _run_sync(self._manager.database.run_in_transaction, _q)

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        def _q(txn):
            rows = list(
                txn.execute_sql(
                    f"SELECT id, entity_type, description, source_id FROM {self._nodes_table} "
                    "WHERE id = @id AND workspace = @ws",
                    params={"id": node_id, "ws": self.workspace},
                    param_types={"id": spanner.param_types.STRING, "ws": spanner.param_types.STRING},
                )
            )
            if not rows:
                return None
            r = rows[0]
            return {"id": r[0], "entity_type": r[1] or "", "description": r[2] or "", "source_id": r[3] or ""}

        return await _run_sync(self._manager.database.run_in_transaction, _q)

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        def _w(txn):
            txn.insert_or_update(
                self._nodes_table,
                columns=["id", "workspace", "entity_type", "description", "source_id"],
                values=[
                    (
                        node_id,
                        self.workspace,
                        node_data.get("entity_type", ""),
                        node_data.get("description", ""),
                        node_data.get("source_id", ""),
                    )
                ],
            )

        await _run_sync(self._manager.database.run_in_transaction, _w)

    async def delete_node(self, node_id: str) -> None:
        def _d(txn):
            # Delete edges first
            txn.execute_update(
                f"DELETE FROM {self._edges_table} WHERE workspace = @ws AND (id = @nid OR target_id = @nid)",
                params={"ws": self.workspace, "nid": node_id},
                param_types={"ws": spanner.param_types.STRING, "nid": spanner.param_types.STRING},
            )
            keyset = spanner.KeySet(keys=[[node_id, self.workspace]])
            txn.delete(self._nodes_table, keyset)

        await _run_sync(self._manager.database.run_in_transaction, _d)

    async def remove_nodes(self, nodes: list[str]) -> None:
        if not nodes:
            return

        def _d(txn):
            # Delete edges referencing any of these nodes
            for nid in nodes:
                txn.execute_update(
                    f"DELETE FROM {self._edges_table} WHERE workspace = @ws AND (id = @nid OR target_id = @nid)",
                    params={"ws": self.workspace, "nid": nid},
                    param_types={"ws": spanner.param_types.STRING, "nid": spanner.param_types.STRING},
                )
            keyset = spanner.KeySet(keys=[[nid, self.workspace] for nid in nodes])
            txn.delete(self._nodes_table, keyset)

        await _run_sync(self._manager.database.run_in_transaction, _d)

    # --- edge CRUD ---

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        def _q(txn):
            rows = list(
                txn.execute_sql(
                    f"SELECT 1 FROM {self._edges_table} "
                    "WHERE id = @src AND target_id = @tgt AND workspace = @ws LIMIT 1",
                    params={"src": source_node_id, "tgt": target_node_id, "ws": self.workspace},
                    param_types={
                        "src": spanner.param_types.STRING,
                        "tgt": spanner.param_types.STRING,
                        "ws": spanner.param_types.STRING,
                    },
                )
            )
            return len(rows) > 0

        return await _run_sync(self._manager.database.run_in_transaction, _q)

    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict[str, str] | None:
        def _q(txn):
            rows = list(
                txn.execute_sql(
                    f"SELECT weight, description, keywords, source_id FROM {self._edges_table} "
                    "WHERE id = @src AND target_id = @tgt AND workspace = @ws",
                    params={"src": source_node_id, "tgt": target_node_id, "ws": self.workspace},
                    param_types={
                        "src": spanner.param_types.STRING,
                        "tgt": spanner.param_types.STRING,
                        "ws": spanner.param_types.STRING,
                    },
                )
            )
            if not rows:
                return None
            r = rows[0]
            return {
                "src_id": source_node_id,
                "tgt_id": target_node_id,
                "weight": float(r[0]) if r[0] is not None else 0.0,
                "description": r[1] or "",
                "keywords": r[2] or "",
                "source_id": r[3] or "",
            }

        return await _run_sync(self._manager.database.run_in_transaction, _q)

    async def _ensure_node_exists(self, node_id: str) -> None:
        """Insert a placeholder node only if it does not already exist.

        Uses an atomic INSERT ... WHERE NOT EXISTS DML statement to avoid
        race conditions when multiple concurrent upsert_edge calls try to
        create the same node simultaneously.
        """
        def _w(txn):
            txn.execute_update(
                f"INSERT INTO {self._nodes_table} (id, workspace, entity_type, description, source_id) "
                "SELECT @id, @ws, '', '', '' "
                f"FROM UNNEST([1]) WHERE NOT EXISTS ("
                f"  SELECT 1 FROM {self._nodes_table} WHERE id = @id AND workspace = @ws"
                ")",
                params={"id": node_id, "ws": self.workspace},
                param_types={"id": spanner.param_types.STRING, "ws": spanner.param_types.STRING},
            )

        await _run_sync(self._manager.database.run_in_transaction, _w)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        # Ensure both endpoint nodes exist without overwriting existing data
        await self._ensure_node_exists(source_node_id)
        await self._ensure_node_exists(target_node_id)

        weight = edge_data.get("weight", "0")
        try:
            weight_f = float(weight)
        except (ValueError, TypeError):
            weight_f = 0.0

        def _w(txn):
            txn.insert_or_update(
                self._edges_table,
                columns=["id", "target_id", "workspace", "weight", "description", "keywords", "source_id"],
                values=[
                    (
                        source_node_id,
                        target_node_id,
                        self.workspace,
                        weight_f,
                        edge_data.get("description", ""),
                        edge_data.get("keywords", ""),
                        edge_data.get("source_id", ""),
                    )
                ],
            )

        await _run_sync(self._manager.database.run_in_transaction, _w)

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        if not edges:
            return

        def _d(txn):
            keyset = spanner.KeySet(
                keys=[[src, tgt, self.workspace] for src, tgt in edges]
            )
            txn.delete(self._edges_table, keyset)

        await _run_sync(self._manager.database.run_in_transaction, _d)

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        if not await self.has_node(source_node_id):
            return None

        def _q(txn):
            # Edges where this node is source
            rows_out = list(
                txn.execute_sql(
                    f"SELECT id, target_id FROM {self._edges_table} "
                    "WHERE id = @nid AND workspace = @ws",
                    params={"nid": source_node_id, "ws": self.workspace},
                    param_types={"nid": spanner.param_types.STRING, "ws": spanner.param_types.STRING},
                )
            )
            # Edges where this node is target (undirected)
            rows_in = list(
                txn.execute_sql(
                    f"SELECT id, target_id FROM {self._edges_table} "
                    "WHERE target_id = @nid AND workspace = @ws",
                    params={"nid": source_node_id, "ws": self.workspace},
                    param_types={"nid": spanner.param_types.STRING, "ws": spanner.param_types.STRING},
                )
            )
            return rows_out + rows_in

        rows = await _run_sync(self._manager.database.run_in_transaction, _q)
        seen = set()
        result = []
        for r in rows:
            edge = (r[0], r[1])
            if edge not in seen:
                seen.add(edge)
                result.append(edge)
        return result

    # --- degree ---

    async def node_degree(self, node_id: str) -> int:
        def _q(txn):
            rows = list(
                txn.execute_sql(
                    f"SELECT COUNT(*) FROM {self._edges_table} "
                    "WHERE workspace = @ws AND (id = @nid OR target_id = @nid)",
                    params={"ws": self.workspace, "nid": node_id},
                    param_types={"ws": spanner.param_types.STRING, "nid": spanner.param_types.STRING},
                )
            )
            return rows[0][0] if rows else 0

        return await _run_sync(self._manager.database.run_in_transaction, _q)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        src_deg = await self.node_degree(src_id)
        tgt_deg = await self.node_degree(tgt_id)
        return src_deg + tgt_deg

    # --- listing ---

    async def get_all_labels(self) -> list[str]:
        def _q(txn):
            rows = list(
                txn.execute_sql(
                    f"SELECT id FROM {self._nodes_table} WHERE workspace = @ws ORDER BY id",
                    params={"ws": self.workspace},
                    param_types={"ws": spanner.param_types.STRING},
                )
            )
            return [r[0] for r in rows]

        return await _run_sync(self._manager.database.run_in_transaction, _q)

    async def get_all_nodes(self) -> list[dict]:
        def _q(txn):
            rows = list(
                txn.execute_sql(
                    f"SELECT id, entity_type, description, source_id FROM {self._nodes_table} WHERE workspace = @ws",
                    params={"ws": self.workspace},
                    param_types={"ws": spanner.param_types.STRING},
                )
            )
            return [
                {"id": r[0], "entity_type": r[1] or "", "description": r[2] or "", "source_id": r[3] or ""}
                for r in rows
            ]

        return await _run_sync(self._manager.database.run_in_transaction, _q)

    async def get_all_edges(self) -> list[dict]:
        def _q(txn):
            rows = list(
                txn.execute_sql(
                    f"SELECT id, target_id, weight, description, keywords, source_id "
                    f"FROM {self._edges_table} WHERE workspace = @ws",
                    params={"ws": self.workspace},
                    param_types={"ws": spanner.param_types.STRING},
                )
            )
            return [
                {
                    "src_id": r[0],
                    "tgt_id": r[1],
                    "weight": float(r[2]) if r[2] is not None else 0.0,
                    "description": r[3] or "",
                    "keywords": r[4] or "",
                    "source_id": r[5] or "",
                }
                for r in rows
            ]

        return await _run_sync(self._manager.database.run_in_transaction, _q)

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        def _q(txn):
            # Count edges per node (both directions) and order by degree
            rows = list(
                txn.execute_sql(
                    f"""SELECT n.id, (
                        SELECT COUNT(*) FROM {self._edges_table} e
                        WHERE e.workspace = @ws AND (e.id = n.id OR e.target_id = n.id)
                    ) AS degree
                    FROM {self._nodes_table} n
                    WHERE n.workspace = @ws
                    ORDER BY degree DESC
                    LIMIT @lim""",
                    params={"ws": self.workspace, "lim": limit},
                    param_types={
                        "ws": spanner.param_types.STRING,
                        "lim": spanner.param_types.INT64,
                    },
                )
            )
            return [r[0] for r in rows]

        return await _run_sync(self._manager.database.run_in_transaction, _q)

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        pattern = f"%{query}%"

        def _q(txn):
            rows = list(
                txn.execute_sql(
                    f"SELECT id FROM {self._nodes_table} "
                    "WHERE workspace = @ws AND LOWER(id) LIKE LOWER(@pattern) "
                    "LIMIT @lim",
                    params={"ws": self.workspace, "pattern": pattern, "lim": limit},
                    param_types={
                        "ws": spanner.param_types.STRING,
                        "pattern": spanner.param_types.STRING,
                        "lim": spanner.param_types.INT64,
                    },
                )
            )
            return [r[0] for r in rows]

        return await _run_sync(self._manager.database.run_in_transaction, _q)

    # --- knowledge graph (BFS) ---

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 3, max_nodes: int = 1000
    ) -> KnowledgeGraph:
        all_nodes_map: dict[str, dict] = {}
        all_edges_list: list[dict] = []
        is_truncated = False

        # Determine starting nodes
        if node_label == "*":
            start_nodes = await self.get_all_labels()
        else:
            start_nodes = await self.search_labels(node_label, limit=max_nodes)

        if not start_nodes:
            return KnowledgeGraph()

        visited: set[str] = set()
        frontier: list[str] = []

        for nid in start_nodes:
            if len(visited) >= max_nodes:
                is_truncated = True
                break
            visited.add(nid)
            frontier.append(nid)

        for _ in range(max_depth):
            if not frontier:
                break
            next_frontier: list[str] = []
            for nid in frontier:
                edges = await self.get_node_edges(nid)
                if edges is None:
                    continue
                for src, tgt in edges:
                    edge_data = await self.get_edge(src, tgt)
                    if edge_data:
                        all_edges_list.append(edge_data)
                    neighbor = tgt if src == nid else src
                    if neighbor not in visited:
                        if len(visited) >= max_nodes:
                            is_truncated = True
                            break
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
                if is_truncated:
                    break
            frontier = next_frontier
            if is_truncated:
                break

        # Fetch node data
        for nid in visited:
            node = await self.get_node(nid)
            if node:
                all_nodes_map[nid] = node

        kg_nodes = [
            KnowledgeGraphNode(
                id=n["id"],
                labels=[n.get("entity_type", "")],
                properties={k: v for k, v in n.items() if k not in ("id",)},
            )
            for n in all_nodes_map.values()
        ]

        # Deduplicate edges
        seen_edges: set[tuple[str, str]] = set()
        kg_edges = []
        for e in all_edges_list:
            key = (e["src_id"], e["tgt_id"])
            if key in seen_edges:
                continue
            seen_edges.add(key)
            kg_edges.append(
                KnowledgeGraphEdge(
                    id=f"{e['src_id']}->{e['tgt_id']}",
                    type=e.get("keywords", ""),
                    source=e["src_id"],
                    target=e["tgt_id"],
                    properties={k: v for k, v in e.items() if k not in ("src_id", "tgt_id")},
                )
            )

        return KnowledgeGraph(nodes=kg_nodes, edges=kg_edges, is_truncated=is_truncated)

    # --- batch optimizations ---

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        if not node_ids:
            return {}

        def _q(txn):
            rows = list(
                txn.execute_sql(
                    f"SELECT id, entity_type, description, source_id FROM {self._nodes_table} "
                    "WHERE id IN UNNEST(@ids) AND workspace = @ws",
                    params={"ids": node_ids, "ws": self.workspace},
                    param_types={
                        "ids": spanner.param_types.Array(spanner.param_types.STRING),
                        "ws": spanner.param_types.STRING,
                    },
                )
            )
            return {
                r[0]: {"id": r[0], "entity_type": r[1] or "", "description": r[2] or "", "source_id": r[3] or ""}
                for r in rows
            }

        return await _run_sync(self._manager.database.run_in_transaction, _q)

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        if not node_ids:
            return {}

        def _q(txn):
            result = {}
            for nid in node_ids:
                rows = list(
                    txn.execute_sql(
                        f"SELECT COUNT(*) FROM {self._edges_table} "
                        "WHERE workspace = @ws AND (id = @nid OR target_id = @nid)",
                        params={"ws": self.workspace, "nid": nid},
                        param_types={"ws": spanner.param_types.STRING, "nid": spanner.param_types.STRING},
                    )
                )
                result[nid] = rows[0][0] if rows else 0
            return result

        return await _run_sync(self._manager.database.run_in_transaction, _q)

    async def get_edges_batch(self, pairs: list[dict[str, str]]) -> dict[tuple[str, str], dict]:
        if not pairs:
            return {}

        def _q(txn):
            result = {}
            for pair in pairs:
                src = pair["src"]
                tgt = pair["tgt"]
                rows = list(
                    txn.execute_sql(
                        f"SELECT weight, description, keywords, source_id FROM {self._edges_table} "
                        "WHERE id = @src AND target_id = @tgt AND workspace = @ws",
                        params={"src": src, "tgt": tgt, "ws": self.workspace},
                        param_types={
                            "src": spanner.param_types.STRING,
                            "tgt": spanner.param_types.STRING,
                            "ws": spanner.param_types.STRING,
                        },
                    )
                )
                if rows:
                    r = rows[0]
                    result[(src, tgt)] = {
                        "src_id": src,
                        "tgt_id": tgt,
                        "weight": float(r[0]) if r[0] is not None else 0.0,
                        "description": r[1] or "",
                        "keywords": r[2] or "",
                        "source_id": r[3] or "",
                    }
            return result

        return await _run_sync(self._manager.database.run_in_transaction, _q)

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        if not node_ids:
            return {}

        def _q(txn):
            result: dict[str, list[tuple[str, str]]] = {nid: [] for nid in node_ids}
            for nid in node_ids:
                rows = list(
                    txn.execute_sql(
                        f"SELECT id, target_id FROM {self._edges_table} "
                        "WHERE workspace = @ws AND (id = @nid OR target_id = @nid)",
                        params={"ws": self.workspace, "nid": nid},
                        param_types={"ws": spanner.param_types.STRING, "nid": spanner.param_types.STRING},
                    )
                )
                for r in rows:
                    result[nid].append((r[0], r[1]))
            return result

        return await _run_sync(self._manager.database.run_in_transaction, _q)

    async def drop(self) -> dict[str, str]:
        try:

            def _drop(txn):
                txn.execute_update(
                    f"DELETE FROM {self._edges_table} WHERE workspace = @ws",
                    params={"ws": self.workspace},
                    param_types={"ws": spanner.param_types.STRING},
                )
                txn.execute_update(
                    f"DELETE FROM {self._nodes_table} WHERE workspace = @ws",
                    params={"ws": self.workspace},
                    param_types={"ws": spanner.param_types.STRING},
                )

            await _run_sync(self._manager.database.run_in_transaction, _drop)
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error("SpannerGraphStorage.drop failed: %s", e)
            return {"status": "error", "message": str(e)}


# ===================================================================
# DocStatus Storage
# ===================================================================

_DOC_STATUS_COLUMNS = [
    "id",
    "workspace",
    "content_summary",
    "content_length",
    "file_path",
    "status",
    "created_at",
    "updated_at",
    "track_id",
    "chunks_count",
    "chunks_list",
    "error_msg",
    "metadata",
]


def _row_to_doc_status(row) -> DocProcessingStatus:
    """Convert a Spanner row to DocProcessingStatus."""
    chunks_list = []
    if row[10]:
        try:
            chunks_list = json.loads(row[10])
        except (json.JSONDecodeError, TypeError):
            chunks_list = []

    metadata = {}
    if row[12]:
        try:
            metadata = json.loads(row[12])
        except (json.JSONDecodeError, TypeError):
            metadata = {}

    return DocProcessingStatus(
        content_summary=row[2] or "",
        content_length=row[3] or 0,
        file_path=row[4] or "",
        status=DocStatus(row[5]) if row[5] else DocStatus.PENDING,
        created_at=row[6] or "",
        updated_at=row[7] or "",
        track_id=row[8],
        chunks_count=row[9],
        chunks_list=chunks_list,
        error_msg=row[11],
        metadata=metadata,
    )


@final
@dataclass
class SpannerDocStatusStorage(DocStatusStorage):
    _manager: SpannerClientManager | None = field(default=None, repr=False)

    def __post_init__(self):
        cfg = get_spanner_config(self.global_config)
        self._spanner_cfg = cfg
        self._table_name = _make_table_name(self.workspace, "doc_status")

    async def initialize(self):
        if self._manager is None:
            cfg = self._spanner_cfg
            self._manager = await SpannerClientManager.get_instance(
                cfg["project_id"], cfg["instance_id"], cfg["database_id"]
            )

        ddl = [
            f"""CREATE TABLE {self._table_name} (
    id STRING(MAX) NOT NULL,
    workspace STRING(MAX) NOT NULL,
    content_summary STRING(MAX),
    content_length INT64,
    file_path STRING(MAX),
    status STRING(MAX),
    created_at STRING(MAX),
    updated_at STRING(MAX),
    track_id STRING(MAX),
    chunks_count INT64,
    chunks_list STRING(MAX),
    error_msg STRING(MAX),
    metadata STRING(MAX)
) PRIMARY KEY (id, workspace)"""
        ]
        db = self._manager.database
        await _run_sync(_ensure_table, db, self._table_name, ddl)
        logger.info("SpannerDocStatusStorage initialized: %s", self._table_name)

    async def finalize(self):
        if self._manager is not None:
            await SpannerClientManager.release_instance(self._manager)
            self._manager = None

    async def index_done_callback(self) -> None:
        pass

    # --- KV base methods ---

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        cols = ", ".join(_DOC_STATUS_COLUMNS)

        def _q(txn):
            rows = list(
                txn.execute_sql(
                    f"SELECT {cols} FROM {self._table_name} WHERE id = @id AND workspace = @ws",
                    params={"id": id, "ws": self.workspace},
                    param_types={"id": spanner.param_types.STRING, "ws": spanner.param_types.STRING},
                )
            )
            return rows[0] if rows else None

        row = await _run_sync(self._manager.database.run_in_transaction, _q)
        if row is None:
            return None
        doc = _row_to_doc_status(row)
        return {
            "id": row[0],
            **doc.__dict__,
        }

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        cols = ", ".join(_DOC_STATUS_COLUMNS)

        def _q(txn):
            rows = list(
                txn.execute_sql(
                    f"SELECT {cols} FROM {self._table_name} "
                    "WHERE id IN UNNEST(@ids) AND workspace = @ws",
                    params={"ids": ids, "ws": self.workspace},
                    param_types={
                        "ids": spanner.param_types.Array(spanner.param_types.STRING),
                        "ws": spanner.param_types.STRING,
                    },
                )
            )
            return {r[0]: r for r in rows}

        rows_map = await _run_sync(self._manager.database.run_in_transaction, _q)
        results = []
        for doc_id in ids:
            row = rows_map.get(doc_id)
            if row is None:
                results.append(None)
                continue
            doc = _row_to_doc_status(row)
            results.append({"id": row[0], **doc.__dict__})
        return results

    async def filter_keys(self, keys: set[str]) -> set[str]:
        if not keys:
            return set()
        keys_list = list(keys)

        def _q(txn):
            result = txn.execute_sql(
                f"SELECT id FROM {self._table_name} "
                "WHERE id IN UNNEST(@ids) AND workspace = @ws",
                params={"ids": keys_list, "ws": self.workspace},
                param_types={
                    "ids": spanner.param_types.Array(spanner.param_types.STRING),
                    "ws": spanner.param_types.STRING,
                },
            )
            return {r[0] for r in result}

        existing = await _run_sync(self._manager.database.run_in_transaction, _q)
        return keys - existing

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return

        def _w(txn):
            rows = []
            for k, v in data.items():
                chunks_list = v.get("chunks_list")
                if isinstance(chunks_list, list):
                    chunks_list = json.dumps(chunks_list, ensure_ascii=False)
                metadata = v.get("metadata")
                if isinstance(metadata, dict):
                    metadata = json.dumps(metadata, ensure_ascii=False)

                status_val = v.get("status", "")
                if isinstance(status_val, DocStatus):
                    status_val = status_val.value

                rows.append(
                    (
                        k,
                        self.workspace,
                        v.get("content_summary", ""),
                        v.get("content_length", 0),
                        v.get("file_path", ""),
                        status_val,
                        v.get("created_at", ""),
                        v.get("updated_at", ""),
                        v.get("track_id"),
                        v.get("chunks_count"),
                        chunks_list,
                        v.get("error_msg"),
                        metadata,
                    )
                )
            txn.insert_or_update(
                self._table_name,
                columns=_DOC_STATUS_COLUMNS,
                values=rows,
            )

        await _run_sync(self._manager.database.run_in_transaction, _w)

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return

        def _d(txn):
            keyset = spanner.KeySet(keys=[[doc_id, self.workspace] for doc_id in ids])
            txn.delete(self._table_name, keyset)

        await _run_sync(self._manager.database.run_in_transaction, _d)

    async def is_empty(self) -> bool:
        def _q(txn):
            result = txn.execute_sql(
                f"SELECT 1 FROM {self._table_name} WHERE workspace = @ws LIMIT 1",
                params={"ws": self.workspace},
                param_types={"ws": spanner.param_types.STRING},
            )
            return len(list(result)) == 0

        return await _run_sync(self._manager.database.run_in_transaction, _q)

    async def drop(self) -> dict[str, str]:
        try:

            def _drop(txn):
                txn.execute_update(
                    f"DELETE FROM {self._table_name} WHERE workspace = @ws",
                    params={"ws": self.workspace},
                    param_types={"ws": spanner.param_types.STRING},
                )

            await _run_sync(self._manager.database.run_in_transaction, _drop)
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error("SpannerDocStatusStorage.drop failed: %s", e)
            return {"status": "error", "message": str(e)}

    # --- DocStatus-specific methods ---

    async def get_status_counts(self) -> dict[str, int]:
        def _q(txn):
            rows = list(
                txn.execute_sql(
                    f"SELECT status, COUNT(*) FROM {self._table_name} "
                    "WHERE workspace = @ws GROUP BY status",
                    params={"ws": self.workspace},
                    param_types={"ws": spanner.param_types.STRING},
                )
            )
            return {r[0]: r[1] for r in rows}

        return await _run_sync(self._manager.database.run_in_transaction, _q)

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        cols = ", ".join(_DOC_STATUS_COLUMNS)

        def _q(txn):
            rows = list(
                txn.execute_sql(
                    f"SELECT {cols} FROM {self._table_name} "
                    "WHERE workspace = @ws AND status = @status",
                    params={"ws": self.workspace, "status": status.value},
                    param_types={
                        "ws": spanner.param_types.STRING,
                        "status": spanner.param_types.STRING,
                    },
                )
            )
            return rows

        rows = await _run_sync(self._manager.database.run_in_transaction, _q)
        result = {}
        for row in rows:
            try:
                result[row[0]] = _row_to_doc_status(row)
            except Exception as e:
                logger.error("Error parsing doc status for %s: %s", row[0], e)
        return result

    async def get_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        cols = ", ".join(_DOC_STATUS_COLUMNS)

        def _q(txn):
            rows = list(
                txn.execute_sql(
                    f"SELECT {cols} FROM {self._table_name} "
                    "WHERE workspace = @ws AND track_id = @tid",
                    params={"ws": self.workspace, "tid": track_id},
                    param_types={
                        "ws": spanner.param_types.STRING,
                        "tid": spanner.param_types.STRING,
                    },
                )
            )
            return rows

        rows = await _run_sync(self._manager.database.run_in_transaction, _q)
        result = {}
        for row in rows:
            try:
                result[row[0]] = _row_to_doc_status(row)
            except Exception as e:
                logger.error("Error parsing doc status for %s: %s", row[0], e)
        return result

    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        # Validate sort
        allowed_sort_fields = {"created_at", "updated_at", "id"}
        if sort_field not in allowed_sort_fields:
            sort_field = "updated_at"
        direction = "DESC" if sort_direction.lower() == "desc" else "ASC"
        offset = (max(1, page) - 1) * page_size
        cols = ", ".join(_DOC_STATUS_COLUMNS)

        def _q(txn):
            # Count query
            count_sql = f"SELECT COUNT(*) FROM {self._table_name} WHERE workspace = @ws"
            params: dict[str, Any] = {"ws": self.workspace}
            ptypes: dict[str, Any] = {"ws": spanner.param_types.STRING}
            if status_filter is not None:
                count_sql += " AND status = @status"
                params["status"] = status_filter.value
                ptypes["status"] = spanner.param_types.STRING

            total = list(txn.execute_sql(count_sql, params=params, param_types=ptypes))
            total_count = total[0][0] if total else 0

            # Data query
            data_sql = (
                f"SELECT {cols} FROM {self._table_name} WHERE workspace = @ws"
            )
            if status_filter is not None:
                data_sql += " AND status = @status"
            data_sql += f" ORDER BY {sort_field} {direction} LIMIT @lim OFFSET @off"
            params["lim"] = page_size
            params["off"] = offset
            ptypes["lim"] = spanner.param_types.INT64
            ptypes["off"] = spanner.param_types.INT64

            rows = list(txn.execute_sql(data_sql, params=params, param_types=ptypes))
            return rows, total_count

        rows, total_count = await _run_sync(self._manager.database.run_in_transaction, _q)
        docs = []
        for row in rows:
            try:
                docs.append((row[0], _row_to_doc_status(row)))
            except Exception as e:
                logger.error("Error parsing doc status for %s: %s", row[0], e)
        return docs, total_count

    async def get_all_status_counts(self) -> dict[str, int]:
        def _q(txn):
            rows = list(
                txn.execute_sql(
                    f"SELECT status, COUNT(*) FROM {self._table_name} GROUP BY status"
                )
            )
            return {r[0]: r[1] for r in rows}

        return await _run_sync(self._manager.database.run_in_transaction, _q)

    async def get_doc_by_file_path(self, file_path: str) -> dict[str, Any] | None:
        cols = ", ".join(_DOC_STATUS_COLUMNS)

        def _q(txn):
            rows = list(
                txn.execute_sql(
                    f"SELECT {cols} FROM {self._table_name} "
                    "WHERE workspace = @ws AND file_path = @fp LIMIT 1",
                    params={"ws": self.workspace, "fp": file_path},
                    param_types={
                        "ws": spanner.param_types.STRING,
                        "fp": spanner.param_types.STRING,
                    },
                )
            )
            return rows[0] if rows else None

        row = await _run_sync(self._manager.database.run_in_transaction, _q)
        if row is None:
            return None
        doc = _row_to_doc_status(row)
        return {"id": row[0], **doc.__dict__}
