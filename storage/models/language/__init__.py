"""
schemas/__init__.py - Persian Language Database Schemas

This package contains all database schemas for the Persian language module.
"""

from .persian.db.schemas import timescaledb
from .persian.db.schemas import clickhouse
from .persian.db.schemas import milvus


__all__ = ['timescaledb', 'clickhouse', 'milvus']
