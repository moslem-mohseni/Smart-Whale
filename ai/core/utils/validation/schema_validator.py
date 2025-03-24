import json
import logging
from jsonschema import validate, ValidationError

class SchemaValidator:
    def __init__(self):
        """
        اعتبارسنجی داده‌های JSON بر اساس Schema
        """
        self.logger = logging.getLogger("SchemaValidator")

    def validate(self, data: dict, schema: dict) -> bool:
        """ بررسی صحت داده‌های JSON بر اساس Schema داده‌شده """
        try:
            validate(instance=data, schema=schema)
            return True
        except ValidationError as e:
            self.logger.error(f"❌ خطای اعتبارسنجی JSON: {e.message}")
            return False
