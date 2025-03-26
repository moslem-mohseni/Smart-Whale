"""
clickhouse/utils/__init__.py - ClickHouse schemas for utility tables

This module contains utility tables used across the Persian language module.
"""

from .system_metrics_table import SystemMetricsTable

__all__ = ['SystemMetricsTable']
