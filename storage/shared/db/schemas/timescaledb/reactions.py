"""
reactions.py - Schema for message reactions

This table stores user reactions to messages, supporting various reaction types
including emoji reactions and custom reactions.
"""

from storage.scripts.base_schema import TimescaleDBSchema


class ReactionsTable(TimescaleDBSchema):
    """
    Table schema for message reactions
    """

    @property
    def name(self) -> str:
        return "reactions"

    @property
    def description(self) -> str:
        return "User reactions to messages"

    @property
    def schema_name(self) -> str:
        return "public"

    def get_create_statement(self) -> str:
        return f"""
        -- Create schema if it doesn't exist
        CREATE SCHEMA IF NOT EXISTS {self.schema_name};

        -- Reactions table
        CREATE TABLE IF NOT EXISTS {self.schema_name}.{self.name} (
            id SERIAL PRIMARY KEY,
            message_id VARCHAR(100) NOT NULL,
            user_id INTEGER NOT NULL,
            reaction_type VARCHAR(50) NOT NULL, -- 'like', 'love', 'laugh', 'wow', 'sad', 'angry', 'custom'
            reaction_content VARCHAR(50), -- for emoji reactions or custom content
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT fk_message
                FOREIGN KEY(message_id)
                REFERENCES {self.schema_name}.messages(message_id)
                ON DELETE CASCADE,
            CONSTRAINT fk_user
                FOREIGN KEY(user_id)
                REFERENCES {self.schema_name}.users(id)
                ON DELETE CASCADE,
            CONSTRAINT unique_reaction UNIQUE(message_id, user_id, reaction_type)
        );

        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_{self.name}_message_id
        ON {self.schema_name}.{self.name} (message_id);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_user_id
        ON {self.schema_name}.{self.name} (user_id);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_reaction_type
        ON {self.schema_name}.{self.name} (reaction_type);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_created_at
        ON {self.schema_name}.{self.name} (created_at);

        -- Add table comment
        COMMENT ON TABLE {self.schema_name}.{self.name} 
        IS 'User reactions to messages';
        """

    def get_check_exists_statement(self) -> str:
        return f"""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = '{self.schema_name}'
            AND table_name = '{self.name}'
        );
        """
