"""
user_subscriptions.py - Schema for user subscription management

This table tracks user subscription information, including subscription type,
payment details, and subscription status.
"""

from storage.scripts.base_schema import TimescaleDBSchema


class UserSubscriptionsTable(TimescaleDBSchema):
    """
    Table schema for user subscriptions
    """

    @property
    def name(self) -> str:
        return "user_subscriptions"

    @property
    def description(self) -> str:
        return "User subscription information and payment details"

    @property
    def schema_name(self) -> str:
        return "public"

    def get_create_statement(self) -> str:
        return f"""
        -- Create schema if it doesn't exist
        CREATE SCHEMA IF NOT EXISTS {self.schema_name};

        -- User subscriptions table
        CREATE TABLE IF NOT EXISTS {self.schema_name}.{self.name} (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            subscription_type VARCHAR(50) NOT NULL, -- 'free', 'pro', 'premium'
            start_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            end_date TIMESTAMPTZ,
            payment_id VARCHAR(255),
            status VARCHAR(50) NOT NULL DEFAULT 'active', -- 'active', 'canceled', 'expired'
            features JSONB DEFAULT '{{}}'::JSONB, -- ویژگی‌های فعال در این اشتراک
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            metadata JSONB DEFAULT '{{}}'::JSONB,
            CONSTRAINT fk_user
                FOREIGN KEY(user_id)
                REFERENCES {self.schema_name}.users(id)
                ON DELETE CASCADE
        );

        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_{self.name}_user_id
        ON {self.schema_name}.{self.name} (user_id);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_subscription_type
        ON {self.schema_name}.{self.name} (subscription_type);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_status
        ON {self.schema_name}.{self.name} (status);

        CREATE INDEX IF NOT EXISTS idx_{self.name}_end_date
        ON {self.schema_name}.{self.name} (end_date);

        -- Add table comment
        COMMENT ON TABLE {self.schema_name}.{self.name} 
        IS 'User subscription information and payment details';
        """

    def get_check_exists_statement(self) -> str:
        return f"""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = '{self.schema_name}'
            AND table_name = '{self.name}'
        );
        """
    