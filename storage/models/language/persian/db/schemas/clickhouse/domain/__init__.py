"""
clickhouse/domain/__init__.py - ClickHouse schemas for domain knowledge usage statistics

This module contains the table schemas for tracking domain knowledge usage and performance metrics.
"""

from .domain_usage_stats_table import DomainUsageStatsTable

__all__ = ['DomainUsageStatsTable']
