import logging
import json
import requests
from datetime import datetime


class ElasticLogHandler:
    def __init__(self, elastic_url, index="system_logs", log_level=logging.INFO):
        """
        ارسال لاگ‌های سیستم به Elasticsearch
        :param elastic_url: آدرس سرور Elasticsearch (مثلاً: "http://localhost:9200")
        :param index: نام Index برای ذخیره لاگ‌ها
        :param log_level: سطح لاگ (پیش‌فرض: INFO)
        """
        self.logger = logging.getLogger("ElasticLogger")
        self.logger.setLevel(log_level)
        self.elastic_url = f"{elastic_url}/{index}/_doc"
        self.index = index

    def log(self, level, message):
        """ ارسال لاگ به Elasticsearch """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level.upper(),
            "message": message,
            "logger": "ElasticLogger"
        }

        try:
            response = requests.post(self.elastic_url, json=log_entry, headers={"Content-Type": "application/json"})
            if response.status_code in [200, 201]:
                self.logger.info(f"✅ لاگ به Elasticsearch ارسال شد → Index: {self.index}")
            else:
                self.logger.error(f"⚠️ خطا در ارسال لاگ به Elasticsearch: {response.status_code}, {response.text}")
        except Exception as e:
            self.logger.error(f"❌ خطا در ارسال لاگ به Elasticsearch: {e}")

    def info(self, message):
        """ ارسال لاگ در سطح INFO به Elasticsearch """
        self.log("info", message)

    def error(self, message):
        """ ارسال لاگ در سطح ERROR به Elasticsearch """
        self.log("error", message)
