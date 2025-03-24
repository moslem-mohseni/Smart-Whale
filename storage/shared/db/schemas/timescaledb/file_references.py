"""
file_references.py - Schema for file references in messages

This table manages the relationship between messages and files, supporting
various reference types (attachments, inline images, thumbnails, etc.).
"""

from storage.scripts.base_schema import TimescaleDBSchema


class FileReferencesTable(TimescaleDBSchema):
    """
    Table schema for file references in messages
    """

    @property
    def name(self) -> str:
        return "file_references"

    @property
    def description(self) -> str:
        return "References between messages and files"

    @property
    def schema_name(self) -> str:
        return "public"

    def get_create_statement(self) -> str:
        return f"""
        -- Create schema if it doesn't exist
        CREATE SCHEMA IF NOT EXISTS {self.schema_name};

        -- File references table
        CREATE TABLE IF NOT EXISTS {self.schema_name}.{self.name} (
            id SERIAL PRIMARY KEY,
            message_id VARCHAR(100) NOT NULL,
            file_id VARCHAR(100) NOT NULL,
            reference_type VARCHAR(50) NOT NULL DEFAULT 'attachment', -- 'attachment', 'inline', 'thumbnail'
            display_order INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT fk_message
                FOREIGN KEY(message_id)
                REFERENCES {self.schema_name}.messages(message_id)
                ON DELETE CASCADE,
            CONSTRAINT fk_file
                FOREIGN KEY(file_id)
                REFERENCES {self.schema_name}.files(file_id)
                ON DELETE CASCADE
        );

        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_{self.name}_message_id
        ON {self.schema_name}.{self.name} (message_id);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_file_id
        ON {self.schema_name}.{self.name} (file_id);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_reference_type
        ON {self.schema_name}.{self.name} (reference_type);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_created_at
        ON {self.schema_name}.{self.name} (created_at);

        -- Add table comment
        COMMENT ON TABLE {self.schema_name}.{self.name} 
        IS 'References between messages and files';
        """

    def get_check_exists_statement(self) -> str:
        return f"""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = '{self.schema_name}'
            AND table_name = '{self.name}'
        );
        """
    