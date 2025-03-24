"""
schemas/__init__.py - Persian Language Database Schemas

This package contains all database schemas for the Persian language module.
"""

from .schemas import timescaledb
from .schemas import clickhouse
from .schemas import milvus


__all__ = ['timescaledb', 'clickhouse', 'milvus']
