"""Spanner client management and common helpers for lightrag-spanner."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from google.cloud import spanner
from google.cloud.spanner_v1.database import Database

logger = logging.getLogger("lightrag.spanner")


class SpannerClientManager:
    """Singleton Spanner client manager for connection reuse.

    Each unique (project, instance, database) combination gets a single
    shared client / database handle, avoiding redundant connections across
    multiple storage classes.
    """

    _instances: dict[str, SpannerClientManager] = {}
    _lock = asyncio.Lock()

    def __init__(self, project_id: str, instance_id: str, database_id: str):
        self._client = spanner.Client(
            project=project_id, disable_builtin_metrics=True
        )
        self._instance = self._client.instance(instance_id)
        self._database: Database = self._instance.database(database_id)
        self._ref_count = 0

    @classmethod
    async def get_instance(
        cls, project_id: str, instance_id: str, database_id: str
    ) -> SpannerClientManager:
        async with cls._lock:
            key = f"{project_id}:{instance_id}:{database_id}"
            if key not in cls._instances:
                cls._instances[key] = cls(project_id, instance_id, database_id)
            cls._instances[key]._ref_count += 1
            return cls._instances[key]

    @classmethod
    async def release_instance(cls, instance: SpannerClientManager) -> None:
        async with cls._lock:
            key = f"{instance.project_id}:{instance.instance_id}:{instance.database_id}"
            if key in cls._instances:
                cls._instances[key]._ref_count -= 1
                if cls._instances[key]._ref_count <= 0:
                    cls._instances[key]._client.close()
                    del cls._instances[key]
                    logger.info("Closed Spanner client for %s", key)

    @property
    def database(self) -> Database:
        return self._database

    @property
    def project_id(self) -> str:
        return self._client.project

    @property
    def instance_id(self) -> str:
        return self._instance.instance_id

    @property
    def database_id(self) -> str:
        return self._database.database_id


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def get_spanner_config(global_config: dict[str, Any]) -> dict[str, str]:
    """Extract Spanner configuration from global_config with env-var fallback.

    Priority: global_config["addon_params"] > environment variables.
    """
    addon = global_config.get("addon_params", {})
    return {
        "project_id": addon.get(
            "spanner_project_id",
            os.environ.get("SPANNER_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
        ),
        "instance_id": addon.get(
            "spanner_instance_id", os.environ.get("SPANNER_INSTANCE", "")
        ),
        "database_id": addon.get(
            "spanner_database_id", os.environ.get("SPANNER_DATABASE", "")
        ),
        "graph_name": addon.get(
            "spanner_graph_name", os.environ.get("SPANNER_GRAPH_NAME", "lightrag_knowledge_graph")
        ),
    }


# ---------------------------------------------------------------------------
# Schema / DDL helpers
# ---------------------------------------------------------------------------

def _table_exists(database: Database, table_name: str) -> bool:
    """Check whether *table_name* exists via INFORMATION_SCHEMA."""
    with database.snapshot() as snapshot:
        results = snapshot.execute_sql(
            "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES "
            "WHERE TABLE_NAME = @table_name",
            params={"table_name": table_name},
            param_types={"table_name": spanner.param_types.STRING},
        )
        row = list(results)
        return row[0][0] > 0 if row else False


def _ensure_table(database: Database, table_name: str, ddl_statements: list[str]) -> None:
    """Create the table described by *ddl_statements* if it does not exist."""
    if _table_exists(database, table_name):
        logger.debug("Table %s already exists, skipping DDL.", table_name)
        return

    operation = database.update_ddl(ddl_statements)
    logger.info("Creating table %s …", table_name)
    operation.result()  # block until DDL completes
    logger.info("Table %s created.", table_name)
