"""
Auto-generated __init__.py file
"""
"""
clickhouse/__init__.py - Persian Language ClickHouse Schemas

This package contains all ClickHouse table schemas used by the Persian language module.
Each submodule defines tables for specific components of the language processing system.
"""

# Import all schema classes so they can be discovered by the schema manager
from .analyzer.analysis_metrics_table import AnalysisMetricsTable
from .dialects.dialect_usage_stats_table import DialectUsageStatsTable
from .domain.domain_usage_stats_table import DomainUsageStatsTable
from .literature.literature_stats_table import LiteratureStatsTable
from .semantic.semantic_usage_stats_table import SemanticUsageStatsTable
from .utils.system_metrics_table import SystemMetricsTable

# List all schema classes for discovery
__all__ = [
    'AnalysisMetricsTable',
    'DialectUsageStatsTable',
    'DomainUsageStatsTable',
    'LiteratureStatsTable',
    'SemanticUsageStatsTable',
    'SystemMetricsTable',
]
