"""
clickhouse/dialects/__init__.py - ClickHouse schemas for Persian dialect usage statistics

This module contains the table schemas for tracking dialect detection, usage, and performance.
"""

from .dialect_usage_stats_table import DialectUsageStatsTable

__all__ = ['DialectUsageStatsTable']
