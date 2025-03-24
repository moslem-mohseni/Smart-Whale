"""
schemas/__init__.py - Persian Language Database Schemas

This package contains all database schemas for the Persian language module.
"""

from .db.schemas import timescaledb
from .db.schemas import clickhouse
from .db.schemas import milvus


__all__ = ['timescaledb', 'clickhouse', 'milvus']
