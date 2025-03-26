"""
domain_usage_stats_table.py - Schema for domain knowledge usage statistics

This table stores statistics about domain knowledge detection and usage, allowing us to track
which domains are most common, concept usage, and performance metrics.
"""

from storage.scripts.base_schema import ClickHouseSchema


class DomainUsageStatsTable(ClickHouseSchema):
    """
    Table schema for domain knowledge usage statistics
    """

    @property
    def name(self) -> str:
        return "domain_usage_stats"

    @property
    def description(self) -> str:
        return "Usage statistics for Persian domain knowledge processing"

    @property
    def database_name(self) -> str:
        return "persian_language"

    def get_create_statement(self) -> str:
        return f"""
        -- Create database if it doesn't exist
        CREATE DATABASE IF NOT EXISTS {self.database_name};

        -- Domain usage statistics table
        CREATE TABLE IF NOT EXISTS {self.database_name}.{self.name} (
            date Date,
            domain_id String,
            domain_code LowCardinality(String),
            domain_name LowCardinality(String),
            detection_count UInt32,
            concept_usage_count UInt32,
            relation_usage_count UInt32,
            attribute_usage_count UInt32,
            avg_confidence Float32,
            avg_processing_time_ms Float32,
            cache_hits UInt32,
            cache_misses UInt32,
            model_invocations UInt32,
            new_concepts_discovered UInt16,
            new_relations_discovered UInt16,
            new_attributes_discovered UInt16,
            errors_count UInt16
        ) ENGINE = SummingMergeTree()
        PARTITION BY toYYYYMM(date)
        ORDER BY (date, domain_id)
        SETTINGS index_granularity = 8192;
        
        -- Add table comment
        COMMENT ON TABLE {self.database_name}.{self.name} IS 'This table tracks usage statistics for Persian domain knowledge processing';
        """
