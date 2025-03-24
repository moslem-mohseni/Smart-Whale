"""
schema_version.py - Schema for schema version tracking

This table stores information about database schema versions and migration history,
allowing the system to track schema changes over time.
"""

from storage.scripts.base_schema import TimescaleDBSchema


class SchemaVersionTable(TimescaleDBSchema):
    """
    Table schema for tracking database schema versions
    """

    @property
    def name(self) -> str:
        return "schema_version"

    @property
    def description(self) -> str:
        return "Tracks database schema versions and migration history"

    @property
    def schema_name(self) -> str:
        return "public"

    def get_create_statement(self) -> str:
        return f"""
        -- Create schema if it doesn't exist
        CREATE SCHEMA IF NOT EXISTS {self.schema_name};

        -- Schema version table
        CREATE TABLE IF NOT EXISTS {self.schema_name}.{self.name} (
            id SERIAL PRIMARY KEY,
            table_name VARCHAR(255) NOT NULL UNIQUE,
            version VARCHAR(50) NOT NULL,
            description TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );

        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_{self.name}_table_name 
        ON {self.schema_name}.{self.name} (table_name);

        -- Add table comment
        COMMENT ON TABLE {self.schema_name}.{self.name} 
        IS 'Tracks database schema versions and migration history';
        """

    def get_check_exists_statement(self) -> str:
        return f"""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = '{self.schema_name}'
            AND table_name = '{self.name}'
        );
        """
    