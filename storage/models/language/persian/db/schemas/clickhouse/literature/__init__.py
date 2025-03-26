"""
clickhouse/literature/__init__.py - ClickHouse schemas for literature analysis statistics

This module contains the table schemas for tracking literature analysis usage and performance metrics.
"""

from .literature_stats_table import LiteratureStatsTable

__all__ = ['LiteratureStatsTable']
