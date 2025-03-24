import os
import json
import logging

# تنظیمات لاگ‌ها
logging.basicConfig(filename="migration_versions.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

VERSION_FILE = "migration_versions.json"


class MigrationVersions:
    """مدیریت نسخه‌های مهاجرت در Milvus"""

    def __init__(self):
        self.version_data = self._load_versions()

    def _load_versions(self):
        """بارگذاری اطلاعات نسخه‌های قبلی از فایل"""
        if os.path.exists(VERSION_FILE):
            with open(VERSION_FILE, "r") as file:
                return json.load(file)
        return {}

    def save_versions(self):
        """ذخیره نسخه‌های جدید در فایل"""
        with open(VERSION_FILE, "w") as file:
            json.dump(self.version_data, file, indent=4)
        logging.info("✅ نسخه‌های جدید مهاجرت ذخیره شدند.")

    def get_current_version(self, collection_name: str):
        """دریافت نسخه فعلی یک Collection"""
        return self.version_data.get(collection_name, "No version recorded")

    def set_version(self, collection_name: str, version: str):
        """تنظیم نسخه جدید برای یک Collection"""
        self.version_data[collection_name] = version
        self.save_versions()
        logging.info(f"🔄 نسخه جدید '{version}' برای Collection '{collection_name}' تنظیم شد.")
        print(f"✅ نسخه جدید '{version}' برای Collection '{collection_name}' تنظیم شد.")

    def list_versions(self):
        """نمایش تمام نسخه‌های ثبت‌شده"""
        print("📜 نسخه‌های ثبت‌شده مهاجرت‌ها:")
        for collection, version in self.version_data.items():
            print(f"🔹 {collection}: {version}")

    def reset_versions(self):
        """حذف تمام نسخه‌های ثبت‌شده"""
        self.version_data = {}
        self.save_versions()
        logging.warning("⚠️ تمام نسخه‌های ثبت‌شده مهاجرت حذف شدند.")
        print("⚠️ تمام نسخه‌های ثبت‌شده مهاجرت حذف شدند.")
