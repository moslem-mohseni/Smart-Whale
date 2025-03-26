"""
clickhouse/semantic/__init__.py - ClickHouse schemas for semantic analysis usage statistics

This module contains the table schemas for tracking semantic analysis operations and performance metrics.
"""

from .semantic_usage_stats_table import SemanticUsageStatsTable

__all__ = ['SemanticUsageStatsTable']
