"""
schemas/__init__.py - Persian Language Database Schemas

This package contains all database schemas for the Persian language module.
"""

from . import timescaledb
from . import clickhouse
from . import milvus

__all__ = ['timescaledb', 'clickhouse', 'milvus']
