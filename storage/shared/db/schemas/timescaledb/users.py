"""
users.py - Schema for user management

This table stores user account information, including authentication details,
profile information, and account settings.
"""

from storage.scripts.base_schema import TimescaleDBSchema


class UsersTable(TimescaleDBSchema):
    """
    Table schema for user accounts
    """

    @property
    def name(self) -> str:
        return "users"

    @property
    def description(self) -> str:
        return "User account information and profile data"

    @property
    def schema_name(self) -> str:
        return "public"

    def get_create_statement(self) -> str:
        return f"""
        -- Create schema if it doesn't exist
        CREATE SCHEMA IF NOT EXISTS {self.schema_name};

        -- Users table
        CREATE TABLE IF NOT EXISTS {self.schema_name}.{self.name} (
            id SERIAL PRIMARY KEY,
            username VARCHAR(100) NOT NULL UNIQUE,
            email VARCHAR(255) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            full_name VARCHAR(255),
            profile_picture VARCHAR(255),
            bio TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_login_at TIMESTAMPTZ,
            is_active BOOLEAN DEFAULT TRUE,
            is_admin BOOLEAN DEFAULT FALSE,
            language_preference VARCHAR(20) DEFAULT 'en',
            notification_settings JSONB DEFAULT '{{}}'::JSONB,
            metadata JSONB DEFAULT '{{}}'::JSONB
        );

        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_{self.name}_username
        ON {self.schema_name}.{self.name} (username);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_email
        ON {self.schema_name}.{self.name} (email);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_is_active
        ON {self.schema_name}.{self.name} (is_active);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_created_at
        ON {self.schema_name}.{self.name} (created_at);

        -- Add table comment
        COMMENT ON TABLE {self.schema_name}.{self.name} 
        IS 'User account information and profile data';
        """

    def get_check_exists_statement(self) -> str:
        return f"""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = '{self.schema_name}'
            AND table_name = '{self.name}'
        );
        """
    