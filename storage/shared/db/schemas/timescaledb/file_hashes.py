"""
file_hashes.py - Schema for file hash management

This table stores unique file hashes to support deduplication and efficient file management,
tracking the relationship between content hashes and multiple file instances.
"""

from storage.scripts.base_schema import TimescaleDBSchema


class FileHashesTable(TimescaleDBSchema):
    """
    Table schema for file hashes
    """

    @property
    def name(self) -> str:
        return "file_hashes"

    @property
    def description(self) -> str:
        return "Unique file hashes for deduplication"

    @property
    def schema_name(self) -> str:
        return "public"

    def get_create_statement(self) -> str:
        return f"""
        -- Create schema if it doesn't exist
        CREATE SCHEMA IF NOT EXISTS {self.schema_name};

        -- File hashes table
        CREATE TABLE IF NOT EXISTS {self.schema_name}.{self.name} (
            id SERIAL PRIMARY KEY,
            hash_id VARCHAR(100) NOT NULL UNIQUE,
            file_hash VARCHAR(255) NOT NULL UNIQUE, -- SHA-256 or other hash
            content_type VARCHAR(100),
            file_count INTEGER DEFAULT 1, -- number of files with this hash
            total_size BIGINT DEFAULT 0, -- sum of file sizes
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            metadata JSONB DEFAULT '{{}}'::JSONB
        );

        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_{self.name}_hash_id
        ON {self.schema_name}.{self.name} (hash_id);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_file_hash
        ON {self.schema_name}.{self.name} (file_hash);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_content_type
        ON {self.schema_name}.{self.name} (content_type);

        -- Add table comment
        COMMENT ON TABLE {self.schema_name}.{self.name} 
        IS 'Unique file hashes for deduplication';
        """

    def get_check_exists_statement(self) -> str:
        return f"""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = '{self.schema_name}'
            AND table_name = '{self.name}'
        );
        """
    