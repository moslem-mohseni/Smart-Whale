"""
system_metrics_table.py - Schema for system-wide metrics

This table stores system-wide metrics for the Persian language module,
including resource usage, error rates, and overall performance indicators.
"""

from storage.scripts.base_schema import ClickHouseSchema


class SystemMetricsTable(ClickHouseSchema):
    """
    Table schema for system-wide metrics
    """

    @property
    def name(self) -> str:
        return "system_metrics"

    @property
    def description(self) -> str:
        return "System-wide metrics for the Persian language module"

    @property
    def database_name(self) -> str:
        return "persian_language"

    def get_create_statement(self) -> str:
        return f"""
        -- Create database if it doesn't exist
        CREATE DATABASE IF NOT EXISTS {self.database_name};

        -- System metrics table
        CREATE TABLE IF NOT EXISTS {self.database_name}.{self.name} (
            date Date,
            timestamp DateTime,
            server_name String,
            component LowCardinality(String),
            metric_name LowCardinality(String),
            metric_value Float64,
            metric_unit LowCardinality(String),
            host_cpu_usage Float32,
            host_memory_usage Float32,
            host_disk_usage Float32,
            container_cpu_usage Float32,
            container_memory_usage Float32,
            container_disk_usage Float32,
            request_count UInt32,
            error_count UInt32,
            avg_response_time_ms Float32,
            p50_response_time_ms Float32,
            p90_response_time_ms Float32,
            p99_response_time_ms Float32
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMM(date)
        ORDER BY (date, timestamp, component, metric_name)
        SETTINGS index_granularity = 8192;
        
        -- Add table comment
        COMMENT ON TABLE {self.database_name}.{self.name} IS 'This table tracks system-wide metrics for the Persian language module';
        """
