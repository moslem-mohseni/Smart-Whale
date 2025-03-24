import logging
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility
from ..config.config import config as collection_config

# تنظیمات اولیه لاگ‌ها
logging.basicConfig(filename="migration.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MigrationManager:
    """مدیریت تغییرات در Collectionهای Milvus"""

    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or collection_config.DEFAULT_COLLECTION_NAME

    def create_collection(self):
        """ایجاد Collection جدید در Milvus در صورت عدم وجود"""
        if utility.has_collection(self.collection_name):
            logging.info(f"Collection '{self.collection_name}' از قبل وجود دارد.")
            print(f"✅ Collection '{self.collection_name}' از قبل وجود دارد.")
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=collection_config.VECTOR_DIMENSIONS),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]

        schema = CollectionSchema(fields=fields, description="Collection for storing vectors")
        collection = Collection(name=self.collection_name, schema=schema)

        logging.info(f"Collection '{self.collection_name}' ایجاد شد.")
        print(f"✅ Collection '{self.collection_name}' ایجاد شد.")

    def drop_collection(self):
        """حذف Collection از Milvus"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logging.warning(f"Collection '{self.collection_name}' حذف شد.")
            print(f"🗑️ Collection '{self.collection_name}' حذف شد.")
        else:
            logging.info(f"Collection '{self.collection_name}' وجود ندارد.")
            print(f"⚠️ Collection '{self.collection_name}' وجود ندارد.")

    def add_field(self, field_name: str, field_type: DataType, description: str = ""):
        """افزودن فیلد جدید به Collection (در نسخه‌های بعدی Milvus امکان‌پذیر خواهد شد)"""
        logging.warning(
            f"⚠️ امکان افزودن فیلد '{field_name}"
            f"' به Collection '{self.collection_name}' در نسخه فعلی Milvus وجود ندارد.")
        print(
            f"⚠️ امکان افزودن فیلد '{field_name}"
            f"' به Collection '{self.collection_name}' در نسخه فعلی Milvus وجود ندارد.")

    def migrate(self):
        """اجرای مهاجرت‌های پایگاه داده"""
        logging.info(f"🔄 شروع عملیات مهاجرت برای Collection '{self.collection_name}'")
        print(f"🔄 شروع عملیات مهاجرت برای Collection '{self.collection_name}'")

        self.create_collection()

        # در آینده می‌توانیم تغییرات بیشتری اعمال کنیم (مثلاً افزودن فیلد)
        # self.add_field("new_feature", DataType.FLOAT_VECTOR, "New feature for AI models")

        logging.info(f"✅ مهاجرت برای Collection '{self.collection_name}' تکمیل شد.")
        print(f"✅ مهاجرت برای Collection '{self.collection_name}' تکمیل شد.")
