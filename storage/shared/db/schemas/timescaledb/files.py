"""
files.py - Schema for file management

This table stores file metadata for files uploaded by users, including
file paths, sizes, types, and access information.
"""

from storage.scripts.base_schema import TimescaleDBSchema


class FilesTable(TimescaleDBSchema):
    """
    Table schema for file metadata
    """

    @property
    def name(self) -> str:
        return "files"

    @property
    def description(self) -> str:
        return "File metadata for uploaded user files"

    @property
    def schema_name(self) -> str:
        return "public"

    def get_create_statement(self) -> str:
        return f"""
        -- Create schema if it doesn't exist
        CREATE SCHEMA IF NOT EXISTS {self.schema_name};

        -- Files table
        CREATE TABLE IF NOT EXISTS {self.schema_name}.{self.name} (
            id SERIAL PRIMARY KEY,
            file_id VARCHAR(100) NOT NULL UNIQUE, -- UUID
            user_id INTEGER NOT NULL,
            file_name VARCHAR(255) NOT NULL,
            original_file_name VARCHAR(255) NOT NULL,
            file_path VARCHAR(500) NOT NULL,
            file_size BIGINT NOT NULL, -- size in bytes
            file_type VARCHAR(100) NOT NULL, -- MIME type
            file_extension VARCHAR(20),
            hash_id VARCHAR(100) NOT NULL, -- reference to file_hashes table
            is_public BOOLEAN DEFAULT FALSE,
            is_encrypted BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_accessed_at TIMESTAMPTZ,
            expiration_date TIMESTAMPTZ, -- optional expiration date
            access_count INTEGER DEFAULT 0,
            metadata JSONB DEFAULT '{{}}'::JSONB,
            CONSTRAINT fk_user
                FOREIGN KEY(user_id)
                REFERENCES {self.schema_name}.users(id)
                ON DELETE CASCADE
        );

        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_{self.name}_file_id
        ON {self.schema_name}.{self.name} (file_id);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_user_id
        ON {self.schema_name}.{self.name} (user_id);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_hash_id
        ON {self.schema_name}.{self.name} (hash_id);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_file_type
        ON {self.schema_name}.{self.name} (file_type);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_created_at
        ON {self.schema_name}.{self.name} (created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_expiration_date
        ON {self.schema_name}.{self.name} (expiration_date);

        -- Add table comment
        COMMENT ON TABLE {self.schema_name}.{self.name} 
        IS 'File metadata for uploaded user files';
        """

    def get_check_exists_statement(self) -> str:
        return f"""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = '{self.schema_name}'
            AND table_name = '{self.name}'
        );
        """
    