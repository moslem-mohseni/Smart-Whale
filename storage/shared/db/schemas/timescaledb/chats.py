"""
chats.py - Schema for chat conversations

This table stores chat conversation metadata, including user ownership,
chat settings, and contextual information.
"""

from storage.scripts.base_schema import TimescaleDBSchema


class ChatsTable(TimescaleDBSchema):
    """
    Table schema for chat conversations
    """

    @property
    def name(self) -> str:
        return "chats"

    @property
    def description(self) -> str:
        return "Chat conversation metadata and settings"

    @property
    def schema_name(self) -> str:
        return "public"

    def get_create_statement(self) -> str:
        return f"""
        -- Create schema if it doesn't exist
        CREATE SCHEMA IF NOT EXISTS {self.schema_name};

        -- Chats table
        CREATE TABLE IF NOT EXISTS {self.schema_name}.{self.name} (
            id SERIAL PRIMARY KEY,
            chat_id VARCHAR(100) NOT NULL UNIQUE, -- UUID
            user_id INTEGER NOT NULL,
            title VARCHAR(255),
            description TEXT,
            chat_type VARCHAR(50) NOT NULL DEFAULT 'personal', -- 'personal', 'group', 'support'
            model_name VARCHAR(100), -- مدل مورد استفاده
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_message_at TIMESTAMPTZ,
            is_archived BOOLEAN DEFAULT FALSE,
            is_pinned BOOLEAN DEFAULT FALSE,
            pin_order INTEGER,
            message_count INTEGER DEFAULT 0,
            settings JSONB DEFAULT '{{}}'::JSONB, -- تنظیمات اختصاصی
            context JSONB DEFAULT '{{}}'::JSONB, -- بافت گفتگو
            metadata JSONB DEFAULT '{{}}'::JSONB,
            CONSTRAINT fk_user
                FOREIGN KEY(user_id)
                REFERENCES {self.schema_name}.users(id)
                ON DELETE CASCADE
        );

        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_{self.name}_chat_id
        ON {self.schema_name}.{self.name} (chat_id);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_user_id
        ON {self.schema_name}.{self.name} (user_id);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_created_at
        ON {self.schema_name}.{self.name} (created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_last_message_at
        ON {self.schema_name}.{self.name} (last_message_at DESC);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_is_archived
        ON {self.schema_name}.{self.name} (is_archived);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_is_pinned
        ON {self.schema_name}.{self.name} (is_pinned, pin_order);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_model_name
        ON {self.schema_name}.{self.name} (model_name);

        -- Add table comment
        COMMENT ON TABLE {self.schema_name}.{self.name} 
        IS 'Chat conversation metadata and settings';
        """

    def get_check_exists_statement(self) -> str:
        return f"""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = '{self.schema_name}'
            AND table_name = '{self.name}'
        );
        """
    