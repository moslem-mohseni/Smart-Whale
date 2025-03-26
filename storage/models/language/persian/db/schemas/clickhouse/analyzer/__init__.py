"""
clickhouse/analyzer/__init__.py - ClickHouse schemas for Persian language analyzer metrics

This module contains the table schemas for storing analyzer performance metrics and analytics.
"""

from .analysis_metrics_table import AnalysisMetricsTable

__all__ = ['AnalysisMetricsTable']
