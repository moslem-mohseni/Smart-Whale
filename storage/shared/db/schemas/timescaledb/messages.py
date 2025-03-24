"""
messages.py - Schema for chat messages

This table stores individual messages within chat conversations, supporting
various content types, threading, and metadata.
"""

from storage.scripts.base_schema import TimescaleDBSchema


class MessagesTable(TimescaleDBSchema):
    """
    Table schema for chat messages
    """

    @property
    def name(self) -> str:
        return "messages"

    @property
    def description(self) -> str:
        return "Individual messages within chat conversations"

    @property
    def schema_name(self) -> str:
        return "public"

    def get_create_statement(self) -> str:
        return f"""
        -- Create schema if it doesn't exist
        CREATE SCHEMA IF NOT EXISTS {self.schema_name};

        -- Messages table
        CREATE TABLE IF NOT EXISTS {self.schema_name}.{self.name} (
            id SERIAL PRIMARY KEY,
            message_id VARCHAR(100) NOT NULL UNIQUE, -- UUID
            chat_id VARCHAR(100) NOT NULL,
            parent_message_id VARCHAR(100), -- for replies
            user_id INTEGER, -- NULL for system messages
            role VARCHAR(50) NOT NULL, -- 'user', 'assistant', 'system'
            content TEXT NOT NULL,
            content_type VARCHAR(50) NOT NULL DEFAULT 'text', -- 'text', 'markdown', 'html', 'image', 'file', 'mixed'
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            is_edited BOOLEAN DEFAULT FALSE,
            is_deleted BOOLEAN DEFAULT FALSE,
            tokens_used INTEGER,
            processing_time_ms INTEGER,
            model_name VARCHAR(100),
            reaction_count INTEGER DEFAULT 0,
            has_files BOOLEAN DEFAULT FALSE,
            metadata JSONB DEFAULT '{{}}'::JSONB,
            CONSTRAINT fk_chat
                FOREIGN KEY(chat_id)
                REFERENCES {self.schema_name}.chats(chat_id)
                ON DELETE CASCADE
        );

        -- Time-based index for time-series capabilities
        SELECT create_hypertable('{self.schema_name}.{self.name}', 'created_at', 
                                if_not_exists => TRUE, 
                                create_default_indexes => FALSE);

        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_{self.name}_message_id
        ON {self.schema_name}.{self.name} (message_id);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_chat_id
        ON {self.schema_name}.{self.name} (chat_id);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_user_id
        ON {self.schema_name}.{self.name} (user_id);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_parent_message_id
        ON {self.schema_name}.{self.name} (parent_message_id);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_role
        ON {self.schema_name}.{self.name} (role);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_content_type
        ON {self.schema_name}.{self.name} (content_type);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_model_name
        ON {self.schema_name}.{self.name} (model_name);

        -- Add table comment
        COMMENT ON TABLE {self.schema_name}.{self.name} 
        IS 'Individual messages within chat conversations';
        """

    def get_check_exists_statement(self) -> str:
        return f"""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = '{self.schema_name}'
            AND table_name = '{self.name}'
        );
        """
    