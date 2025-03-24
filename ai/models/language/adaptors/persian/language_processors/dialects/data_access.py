# language_processors/dialects/data_access.py

"""
ماژول data_access.py

این ماژول مسئول مدیریت دسترسی به داده‌های لهجه (dialects، ویژگی‌ها، واژگان و قواعد تبدیل) است.
ابتدا سعی می‌کند داده‌ها را از کش (Redis) دریافت کند؛ در صورت عدم وجود، از پایگاه داده (ClickHouse) بازیابی کند.
اگر هیچ داده‌ای یافت نشود، داده‌های پیش‌فرض را بارگذاری (initialize) کرده و همزمان در پایگاه داده و کش ذخیره می‌کند.
علاوه بر این، این ماژول از سرویس‌های Milvus، Kafka، Performance Metrics، Health Check و File Management نیز استفاده می‌کند.
"""

import importlib
import json
import logging
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ...config import CONFIG
from ai.models.language.infrastructure.caching.redis_adapter import RedisAdapter
from ai.models.language.infrastructure.caching.cache_manager import CacheManager
from ai.models.language.infrastructure.vector_store.vector_search import VectorSearch
from ai.models.language.infrastructure.vector_store.milvus_adapter import MilvusAdapter
from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB
from ai.models.language.infrastructure.messaging.kafka_producer import KafkaProducer
from ai.models.language.infrastructure.messaging.kafka_consumer import KafkaConsumer
from ai.models.language.infrastructure.monitoring.performance_metrics import PerformanceMetrics
from ai.models.language.infrastructure.monitoring.health_check import HealthCheck
from ai.models.language.infrastructure.file_management.file_service import FileManagementService

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# اگر hazm موجود باشد، تنظیمات اولیه آن را انجام می‌دهیم
try:
    if CONFIG.get("use_hazm", True):
        from hazm import Normalizer, SentenceTokenizer, WordTokenizer
except ImportError:
    logger.warning("کتابخانه hazm یافت نشد. قابلیت‌های پایه محدود خواهند بود.")
    CONFIG["use_hazm"] = False


class DialectDataAccess:
    """
    کلاس DialectDataAccess مسئول مدیریت داده‌های مربوط به لهجه‌ها است.

    وظایف اصلی:
      - اتصال به زیرساخت‌های کش (Redis)، پایگاه داده (ClickHouse)، Milvus، Kafka و سایر سرویس‌ها.
      - ایجاد ساختارهای پایگاه داده (جدول‌های dialects، dialect_features، dialect_words، dialect_conversion_rules، dialect_detection_history و dialect_text_vectors).
      - بارگذاری داده‌های موجود؛ در صورت عدم وجود، بارگذاری داده‌های پیش‌فرض.
      - ارائه توابع ذخیره و به‌روزرسانی داده‌ها.
      - مدیریت کش: ابتدا داده‌ها از کش بازیابی می‌شوند و در صورت نیاز در پایگاه داده ذخیره می‌شوند.
      - فراهم آوردن توابع صادرات و واردسازی (import/export) دانش لهجه‌ای.
    """

    def __init__(self):
        self.logger = logger
        self.logger.info("Initializing DialectDataAccess...")
        self.redis = RedisAdapter()
        self.cache_manager = CacheManager()
        self.database = ClickHouseDB()
        self.vector_store = MilvusAdapter(collection_name="dialect_vectors")
        self.vector_search = VectorSearch()
        self.performance_metrics = PerformanceMetrics()
        self.health_check = HealthCheck()
        self.file_service = FileManagementService()
        self.kafka_producer = KafkaProducer()
        self.kafka_consumer = KafkaConsumer()
        self._setup_kafka_consumer()
        self._setup_database()

        # بارگذاری داده‌های لهجه، ویژگی‌ها، واژگان و قواعد تبدیل
        self.dialects = self._load_dialects()
        self.dialect_features = self._load_dialect_features()
        self.dialect_words = self._load_dialect_words()
        self.dialect_conversion_rules = self._load_conversion_rules()

        # آمار عملکرد
        self.statistics = {
            "requests": 0,
            "cache_hits": 0,
            "teacher_uses": 0,
            "smart_model_uses": 0,
            "new_dialect_features_discovered": 0,
            "new_dialect_words_discovered": 0,
            "detection_count": 0,
            "conversion_count": 0,
            "start_time": time.time()
        }
        self.logger.info("DialectDataAccess initialized successfully.")

    def _setup_database(self):
        """ایجاد ساختارهای پایگاه داده در ClickHouse."""
        try:
            self.database.execute_query("""
            CREATE TABLE IF NOT EXISTS dialects (
                dialect_id String,
                dialect_name String,
                dialect_code String,
                region String,
                description String,
                parent_dialect String DEFAULT '',
                popularity Float32 DEFAULT 0,
                source String,
                discovery_time DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            ORDER BY (dialect_id, discovery_time)
            """)
            self.database.execute_query("""
            CREATE TABLE IF NOT EXISTS dialect_features (
                feature_id String,
                dialect_id String,
                feature_type String,
                feature_pattern String,
                description String,
                examples Array(String),
                confidence Float32,
                source String,
                usage_count UInt32 DEFAULT 1,
                discovery_time DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            ORDER BY (feature_id, dialect_id, discovery_time)
            """)
            self.database.execute_query("""
            CREATE TABLE IF NOT EXISTS dialect_words (
                word_id String,
                dialect_id String,
                word String,
                standard_equivalent String,
                definition String,
                part_of_speech String DEFAULT '',
                usage Array(String),
                confidence Float32,
                source String,
                usage_count UInt32 DEFAULT 1,
                discovery_time DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            ORDER BY (word_id, dialect_id, discovery_time)
            """)
            self.database.execute_query("""
            CREATE TABLE IF NOT EXISTS dialect_conversion_rules (
                rule_id String,
                source_dialect String,
                target_dialect String,
                rule_type String,
                rule_pattern String,
                replacement String,
                description String,
                examples Array(String),
                confidence Float32,
                source String,
                usage_count UInt32 DEFAULT 1,
                discovery_time DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            ORDER BY (rule_id, source_dialect, target_dialect, discovery_time)
            """)
            self.database.execute_query("""
            CREATE TABLE IF NOT EXISTS dialect_detection_history (
                detection_id String,
                text String,
                detected_dialect String,
                confidence Float32,
                dialect_features Array(String),
                detection_time DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            ORDER BY (detection_id, detection_time)
            """)
            self.database.execute_query("""
            CREATE TABLE IF NOT EXISTS dialect_text_vectors (
                text_hash String,
                text String,
                dialect_id String,
                vector Array(Float32),
                timestamp DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            ORDER BY (text_hash, dialect_id, timestamp)
            """)
            self.logger.info("Database structures set up successfully.")
        except Exception as e:
            self.logger.error(f"خطا در تنظیم پایگاه داده: {e}")

    def _setup_kafka_consumer(self):
        """تنظیم مصرف‌کننده Kafka جهت دریافت پیام‌های به‌روزرسانی لهجه."""
        try:
            self.kafka_consumer.subscribe(
                topic="dialect_updates",
                group_id="dialect_processor",
                handler=self._handle_kafka_message
            )
        except Exception as e:
            self.logger.error(f"خطا در تنظیم Kafka Consumer: {e}")

    def _handle_kafka_message(self, message: str):
        """پردازش پیام دریافتی از Kafka برای به‌روزرسانی داده‌ها."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            if msg_type == "new_dialect_feature":
                self._add_dialect_feature_from_message(data)
            elif msg_type == "new_dialect_word":
                self._add_dialect_word_from_message(data)
            elif msg_type == "new_conversion_rule":
                self._add_conversion_rule_from_message(data)
            elif msg_type == "dialect_learning":
                self._learn_from_message(data)
            self.logger.info(f"Kafka message processed: {msg_type}")
        except Exception as e:
            self.logger.error(f"خطا در پردازش پیام Kafka: {e}")

    def _load_dialects(self) -> Dict[str, Dict[str, Any]]:
        """بارگذاری لهجه‌ها از پایگاه داده؛ در صورت عدم وجود، داده‌های پیش‌فرض را بارگذاری می‌کند."""
        try:
            result = self.database.execute_query("SELECT * FROM dialects")
            if result and len(result) > 0:
                dialects = {row["dialect_id"]: row for row in result}
                return dialects
            # اگر هیچ داده‌ای یافت نشد، داده‌های پیش‌فرض را بارگذاری کن
            self._initialize_basic_dialects()
            result = self.database.execute_query("SELECT * FROM dialects")
            return {row["dialect_id"]: row for row in result} if result else {}
        except Exception as e:
            self.logger.error(f"خطا در بارگذاری لهجه‌ها: {e}")
            return {}

    def _load_dialect_features(self) -> Dict[str, Dict[str, Any]]:
        """بارگذاری ویژگی‌های لهجه‌ای از پایگاه داده؛ در صورت عدم وجود، پیش‌فرض‌ها را بارگذاری می‌کند."""
        try:
            result = self.database.execute_query("SELECT * FROM dialect_features")
            if result and len(result) > 0:
                features = {row["feature_id"]: row for row in result}
                self.statistics["new_dialect_features_discovered"] = len(features)
                return features
            self._initialize_basic_dialect_features()
            result = self.database.execute_query("SELECT * FROM dialect_features")
            return {row["feature_id"]: row for row in result} if result else {}
        except Exception as e:
            self.logger.error(f"خطا در بارگذاری ویژگی‌های لهجه‌ای: {e}")
            return {}

    def _load_dialect_words(self) -> Dict[str, Dict[str, Any]]:
        """بارگذاری واژگان لهجه‌ای از پایگاه داده؛ در صورت عدم وجود، پیش‌فرض‌ها را بارگذاری می‌کند."""
        try:
            result = self.database.execute_query("SELECT * FROM dialect_words")
            if result and len(result) > 0:
                words = {row["word_id"]: row for row in result}
                self.statistics["new_dialect_words_discovered"] = len(words)
                return words
            self._initialize_basic_dialect_words()
            result = self.database.execute_query("SELECT * FROM dialect_words")
            return {row["word_id"]: row for row in result} if result else {}
        except Exception as e:
            self.logger.error(f"خطا در بارگذاری واژگان لهجه‌ای: {e}")
            return {}

    def _load_conversion_rules(self) -> Dict[str, Dict[str, Any]]:
        """بارگذاری قواعد تبدیل لهجه‌ای از پایگاه داده؛ در صورت عدم وجود، پیش‌فرض‌ها را بارگذاری می‌کند."""
        try:
            result = self.database.execute_query("SELECT * FROM dialect_conversion_rules")
            if result and len(result) > 0:
                rules = {row["rule_id"]: row for row in result}
                return rules
            self._initialize_basic_conversion_rules()
            result = self.database.execute_query("SELECT * FROM dialect_conversion_rules")
            return {row["rule_id"]: row for row in result} if result else {}
        except Exception as e:
            self.logger.error(f"خطا در بارگذاری قواعد تبدیل لهجه‌ای: {e}")
            return {}

    def _initialize_basic_dialects(self):
        """بارگذاری داده‌های پیش‌فرض لهجه‌ها."""
        basic_dialects = [
            {
                "dialect_id": "d_1001",
                "dialect_name": "فارسی معیار",
                "dialect_code": "STANDARD",
                "region": "سراسر ایران",
                "description": "فارسی رسمی و معیار که در رسانه‌های ملی و آموزش رسمی استفاده می‌شود",
                "parent_dialect": "",
                "popularity": 0.9,
                "source": "default"
            },
            {
                "dialect_id": "d_1002",
                "dialect_name": "تهرانی",
                "dialect_code": "TEHRANI",
                "region": "تهران",
                "description": "لهجه رایج در تهران و اطراف آن که تأثیر زیادی بر فارسی معیار داشته است",
                "parent_dialect": "d_1001",
                "popularity": 0.85,
                "source": "default"
            },
            {
                "dialect_id": "d_1003",
                "dialect_name": "اصفهانی",
                "dialect_code": "ISFAHANI",
                "region": "اصفهان",
                "description": "لهجه مردم اصفهان با ویژگی‌های آوایی خاص و واژگان منحصر به فرد",
                "parent_dialect": "d_1001",
                "popularity": 0.6,
                "source": "default"
            },
            {
                "dialect_id": "d_1004",
                "dialect_name": "مشهدی",
                "dialect_code": "MASHHADI",
                "region": "خراسان",
                "description": "لهجه رایج در مشهد و مناطق خراسان با تأثیرات زبانی از خراسان بزرگ",
                "parent_dialect": "d_1001",
                "popularity": 0.65,
                "source": "default"
            },
            {
                "dialect_id": "d_1005",
                "dialect_name": "شیرازی",
                "dialect_code": "SHIRAZI",
                "region": "فارس",
                "description": "لهجه شیراز و مناطق مرکزی استان فارس با تلفظ متمایز و واژگان خاص",
                "parent_dialect": "d_1001",
                "popularity": 0.6,
                "source": "default"
            },
            {
                "dialect_id": "d_1006",
                "dialect_name": "تبریزی",
                "dialect_code": "TABRIZI",
                "region": "آذربایجان شرقی",
                "description": "لهجه فارسی رایج در تبریز که تحت تأثیر زبان ترکی آذربایجانی است",
                "parent_dialect": "d_1001",
                "popularity": 0.5,
                "source": "default"
            },
            {
                "dialect_id": "d_1007",
                "dialect_name": "گیلکی",
                "dialect_code": "GILAKI",
                "region": "گیلان",
                "description": "زبان/گویش رایج در استان گیلان که به عنوان یکی از زبان‌های ایرانی شمال غربی شناخته می‌شود",
                "parent_dialect": "",
                "popularity": 0.4,
                "source": "default"
            },
            {
                "dialect_id": "d_1008",
                "dialect_name": "مازندرانی",
                "dialect_code": "MAZANI",
                "region": "مازندران",
                "description": "زبان/گویش رایج در استان مازندران که از زبان‌های ایرانی شمال غربی است",
                "parent_dialect": "",
                "popularity": 0.4,
                "source": "default"
            },
            {
                "dialect_id": "d_1009",
                "dialect_name": "لری",
                "dialect_code": "LORI",
                "region": "لرستان و بخش‌هایی از خوزستان و ایلام",
                "description": "زبان/گویش مردم لر با ویژگی‌های آوایی و واژگانی متمایز",
                "parent_dialect": "",
                "popularity": 0.45,
                "source": "default"
            },
            {
                "dialect_id": "d_1010",
                "dialect_name": "کردی",
                "dialect_code": "KURDISH",
                "region": "کردستان، کرمانشاه و مناطق کردنشین",
                "description": "زبان/گویش کردی با گویش‌های متعدد که در مناطق کردنشین ایران رایج است",
                "parent_dialect": "",
                "popularity": 0.5,
                "source": "default"
            }
        ]
        for dialect in basic_dialects:
            self._store_dialect(dialect)

    def _initialize_basic_dialect_features(self):
        """بارگذاری داده‌های پیش‌فرض ویژگی‌های لهجه‌ای."""
        basic_features = [
            # تهرانی
            {
                "feature_id": "f_tehran_1",
                "dialect_id": "d_1002",
                "feature_type": "PHONETIC",
                "feature_pattern": r"\bمی(‌|\s)?(خوا[هم]|رم|زنم|کنم|گم|دم)",
                "description": "تبدیل «می» به «می» کشیده در افعال",
                "examples": ["من میـــخوام برم", "میـــگم که نمیشه"],
                "confidence": 0.9,
                "source": "default"
            },
            {
                "feature_id": "f_tehran_2",
                "dialect_id": "d_1002",
                "feature_type": "VOCABULARY",
                "feature_pattern": r"\b(چاکر[یم]?|نوکر[یم]?|مخلص[یم]?)\b",
                "description": "استفاده از اصطلاحات تعارفی خاص",
                "examples": ["چاکریم", "مخلصم داداش"],
                "confidence": 0.85,
                "source": "default"
            },
            # اصفهانی
            {
                "feature_id": "f_isfahan_1",
                "dialect_id": "d_1003",
                "feature_type": "PHONETIC",
                "feature_pattern": r"\b(بلی|بعله)\b",
                "description": "استفاده از «بلی» به جای «بله»",
                "examples": ["بلی آقا", "بعله، همینطوره"],
                "confidence": 0.9,
                "source": "default"
            },
            {
                "feature_id": "f_isfahan_2",
                "dialect_id": "d_1003",
                "feature_type": "GRAMMATICAL",
                "feature_pattern": r"\b(مَ|مو)\b",
                "description": "استفاده از «مَ» یا «مو» به جای «من»",
                "examples": ["مَ اومدم", "مو نمیدونم"],
                "confidence": 0.85,
                "source": "default"
            },
            # مشهدی
            {
                "feature_id": "f_mashhad_1",
                "dialect_id": "d_1004",
                "feature_type": "PHONETIC",
                "feature_pattern": r"\b(اَ?ری|هَری)\b",
                "description": "استفاده از «اری» یا «هری» به جای «آره» یا «بله»",
                "examples": ["اری، همی جوره", "هری، خودمم"],
                "confidence": 0.9,
                "source": "default"
            },
            {
                "feature_id": "f_mashhad_2",
                "dialect_id": "d_1004",
                "feature_type": "GRAMMATICAL",
                "feature_pattern": r"\bدَر(ُ|و)م\b",
                "description": "استفاده از «دَروم» به جای «دارم»",
                "examples": ["مو دَروم میرم", "خی دَروم میگم"],
                "confidence": 0.85,
                "source": "default"
            },
            # شیرازی
            {
                "feature_id": "f_shiraz_1",
                "dialect_id": "d_1005",
                "feature_type": "PHONETIC",
                "feature_pattern": r"\b(شما|اونا|اینا) (هَ|حَ)[سش]تین\b",
                "description": "تبدیل «هستید» به «هستین»",
                "examples": ["شما هستین؟", "اونا هستین فردا؟"],
                "confidence": 0.9,
                "source": "default"
            },
            {
                "feature_id": "f_shiraz_2",
                "dialect_id": "d_1005",
                "feature_type": "VOCABULARY",
                "feature_pattern": r"\bمَشتی\b",
                "description": "استفاده از واژه «مشتی» در گفتار",
                "examples": ["مشتی کجایی؟", "بیا اینجا مشتی"],
                "confidence": 0.85,
                "source": "default"
            }
        ]
        for feature in basic_features:
            self._store_dialect_feature(feature)

    def _initialize_basic_dialect_words(self):
        """بارگذاری داده‌های پیش‌فرض واژگان لهجه‌ای."""
        basic_words = [
            # تهرانی
            {
                "word_id": "w_tehran_1",
                "dialect_id": "d_1002",
                "word": "داش",
                "standard_equivalent": "برادر",
                "definition": "اصطلاح خودمانی برای اشاره به دوست نزدیک مذکر",
                "part_of_speech": "NOUN",
                "usage": ["داش مشتی", "داش گلم"],
                "confidence": 0.9,
                "source": "default"
            },
            {
                "word_id": "w_tehran_2",
                "dialect_id": "d_1002",
                "word": "چاکرم",
                "standard_equivalent": "مخلص شما هستم",
                "definition": "اصطلاح تعارفی برای ابراز احترام و ارادت",
                "part_of_speech": "EXPRESSION",
                "usage": ["چاکرم داداش", "چاکرتم"],
                "confidence": 0.9,
                "source": "default"
            },
            # اصفهانی
            {
                "word_id": "w_isfahan_1",
                "dialect_id": "d_1003",
                "word": "پاچَنار",
                "standard_equivalent": "باغچه",
                "definition": "باغچه کوچک یا زمین پایین درخت چنار",
                "part_of_speech": "NOUN",
                "usage": ["پاچنار حیاطمون", "برو تو پاچنار"],
                "confidence": 0.9,
                "source": "default"
            },
            {
                "word_id": "w_isfahan_2",
                "dialect_id": "d_1003",
                "word": "چوقول",
                "standard_equivalent": "سخن‌چین",
                "definition": "فردی که درباره دیگران خبرچینی می‌کند",
                "part_of_speech": "NOUN",
                "usage": ["چوقولی نکن", "چوقول محل"],
                "confidence": 0.9,
                "source": "default"
            },
            # مشهدی
            {
                "word_id": "w_mashhad_1",
                "dialect_id": "d_1004",
                "word": "اوغور",
                "standard_equivalent": "آغوش",
                "definition": "آغوش، بغل",
                "part_of_speech": "NOUN",
                "usage": ["بچه ره بنداز تو اوغور", "بیا تو اوغورُم"],
                "confidence": 0.9,
                "source": "default"
            },
            {
                "word_id": "w_mashhad_2",
                "dialect_id": "d_1004",
                "word": "ایزار",
                "standard_equivalent": "ملافه",
                "definition": "ملافه، پارچه بزرگ برای پوشاندن",
                "part_of_speech": "NOUN",
                "usage": ["ایزار تمیز", "ایزارِ سفید"],
                "confidence": 0.9,
                "source": "default"
            },
            # شیرازی
            {
                "word_id": "w_shiraz_1",
                "dialect_id": "d_1005",
                "word": "کاکو",
                "standard_equivalent": "برادر",
                "definition": "اصطلاح خودمانی برای خطاب به برادر یا دوست",
                "part_of_speech": "NOUN",
                "usage": ["کاکو جان", "بیا اینجا کاکو"],
                "confidence": 0.9,
                "source": "default"
            },
            {
                "word_id": "w_shiraz_2",
                "dialect_id": "d_1005",
                "word": "لُکنی",
                "standard_equivalent": "ظرف مسی",
                "definition": "ظرف مسی بزرگ برای پختن غذا",
                "part_of_speech": "NOUN",
                "usage": ["لکنی را بشور", "تو لُکنی آش بپز"],
                "confidence": 0.9,
                "source": "default"
            }
        ]
        for word in basic_words:
            self._store_dialect_word(word)

    def _initialize_basic_conversion_rules(self):
        """بارگذاری داده‌های پیش‌فرض قواعد تبدیل لهجه‌ای."""
        basic_rules = [
            # تهرانی به معیار
            {
                "rule_id": "r_tehran_std_1",
                "source_dialect": "d_1002",
                "target_dialect": "d_1001",
                "rule_type": "PHONETIC",
                "rule_pattern": r"می(ـ+)(خوا[هم]|رم|زنم|کنم|گم|دم)",
                "replacement": r"می‌\2",
                "description": "حذف کشش «می» در افعال تهرانی",
                "examples": ["من میـــخوام برم -> من می‌خواهم بروم", "میـــگم -> می‌گویم"],
                "confidence": 0.9,
                "source": "default"
            },
            {
                "rule_id": "r_tehran_std_2",
                "source_dialect": "d_1002",
                "target_dialect": "d_1001",
                "rule_type": "VOCABULARY",
                "rule_pattern": r"\b(داش|دادا)\b",
                "replacement": r"برادر",
                "description": "تبدیل واژگان محاوره‌ای تهرانی به معیار",
                "examples": ["داش مشتی -> برادر محترم", "دادا گلم -> برادر عزیزم"],
                "confidence": 0.85,
                "source": "default"
            },
            # اصفهانی به معیار
            {
                "rule_id": "r_isfahan_std_1",
                "source_dialect": "d_1003",
                "target_dialect": "d_1001",
                "rule_type": "PHONETIC",
                "rule_pattern": r"\b(بلی|بعله)\b",
                "replacement": r"بله",
                "description": "تبدیل «بلی/بعله» اصفهانی به «بله» معیار",
                "examples": ["بلی آقا -> بله آقا", "بعله، همینطوره -> بله، همینطور است"],
                "confidence": 0.9,
                "source": "default"
            },
            {
                "rule_id": "r_isfahan_std_2",
                "source_dialect": "d_1003",
                "target_dialect": "d_1001",
                "rule_type": "GRAMMATICAL",
                "rule_pattern": r"\b(مَ|مو)\b",
                "replacement": r"من",
                "description": "تبدیل «مَ/مو» اصفهانی به «من» معیار",
                "examples": ["مَ اومدم -> من آمدم", "مو نمیدونم -> من نمی‌دانم"],
                "confidence": 0.9,
                "source": "default"
            },
            # مشهدی به معیار
            {
                "rule_id": "r_mashhad_std_1",
                "source_dialect": "d_1004",
                "target_dialect": "d_1001",
                "rule_type": "PHONETIC",
                "rule_pattern": r"\b(اَ?ری|هَری)\b",
                "replacement": r"بله",
                "description": "تبدیل «اری/هری» مشهدی به «بله» معیار",
                "examples": ["اری، همی جوره -> بله، همینطور است", "هری، خودمم -> بله، خودم هستم"],
                "confidence": 0.9,
                "source": "default"
            },
            {
                "rule_id": "r_mashhad_std_2",
                "source_dialect": "d_1004",
                "target_dialect": "d_1001",
                "rule_type": "GRAMMATICAL",
                "rule_pattern": r"\bدَر(ُ|و)م\b",
                "replacement": r"دارم",
                "description": "تبدیل «دَروم» مشهدی به «دارم» معیار",
                "examples": ["مو دَروم میرم -> من دارم می‌روم", "خی دَروم میگم -> خب دارم می‌گویم"],
                "confidence": 0.9,
                "source": "default"
            }
        ]
        for rule in basic_rules:
            self._store_conversion_rule(rule)

    def _store_dialect(self, dialect: Dict[str, Any]):
        """ذخیره یک رکورد لهجه در پایگاه داده و دیکشنری محلی."""
        try:
            existing = self.database.execute_query(
                f"SELECT dialect_id FROM dialects WHERE dialect_id='{dialect['dialect_id']}' LIMIT 1"
            )
            if not existing or len(existing) == 0:
                self.database.insert_data("dialects", dialect)
        except Exception as e:
            self.logger.error(f"خطا در ذخیره لهجه: {e}")

    def _store_dialect_feature(self, feature: Dict[str, Any]):
        """ذخیره یک رکورد ویژگی لهجه‌ای در پایگاه داده."""
        try:
            existing = self.database.execute_query(
                f"SELECT feature_id FROM dialect_features WHERE feature_id='{feature['feature_id']}' LIMIT 1"
            )
            if not existing or len(existing) == 0:
                self.database.insert_data("dialect_features", feature)
            else:
                self.database.execute_query(f"""
                UPDATE dialect_features
                SET usage_count = usage_count + 1,
                    confidence = {feature['confidence']}
                WHERE feature_id = '{feature['feature_id']}'
                """)
        except Exception as e:
            self.logger.error(f"خطا در ذخیره ویژگی لهجه‌ای: {e}")

    def _store_dialect_word(self, word: Dict[str, Any]):
        """ذخیره یک رکورد واژه لهجه‌ای در پایگاه داده."""
        try:
            existing = self.database.execute_query(
                f"SELECT word_id FROM dialect_words WHERE word_id='{word['word_id']}' LIMIT 1"
            )
            if not existing or len(existing) == 0:
                self.database.insert_data("dialect_words", word)
            else:
                self.database.execute_query(f"""
                UPDATE dialect_words
                SET usage_count = usage_count + 1,
                    confidence = {word['confidence']}
                WHERE word_id = '{word['word_id']}'
                """)
        except Exception as e:
            self.logger.error(f"خطا در ذخیره واژه لهجه‌ای: {e}")

    def _store_conversion_rule(self, rule: Dict[str, Any]):
        """ذخیره یک رکورد قاعده تبدیل لهجه‌ای در پایگاه داده."""
        try:
            existing = self.database.execute_query(
                f"SELECT rule_id FROM dialect_conversion_rules WHERE rule_id='{rule['rule_id']}' LIMIT 1"
            )
            if not existing or len(existing) == 0:
                self.database.insert_data("dialect_conversion_rules", rule)
            else:
                self.database.execute_query(f"""
                UPDATE dialect_conversion_rules
                SET usage_count = usage_count + 1,
                    confidence = {rule['confidence']}
                WHERE rule_id = '{rule['rule_id']}'
                """)
        except Exception as e:
            self.logger.error(f"خطا در ذخیره قاعده تبدیل لهجه‌ای: {e}")

    def export_knowledge(self, file_path: Optional[str] = None) -> str:
        """
        خروجی گرفتن از تمامی دانش لهجه‌ای به صورت JSON.
        اگر مسیر فایل مشخص نباشد، یک نام پیش‌فرض بر مبنای timestamp تولید می‌شود.
        """
        if file_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_path = f"dialect_knowledge_{timestamp}.json"
        export_data = {
            "dialects": list(self.dialects.values()),
            "features": list(self.dialect_features.values()),
            "words": list(self.dialect_words.values()),
            "conversion_rules": list(self.dialect_conversion_rules.values()),
            "statistics": self.get_statistics(),
            "timestamp": time.time()
        }
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            return file_path
        except Exception as e:
            self.logger.error(f"خطا در خروجی گرفتن: {e}")
            raise

    def import_knowledge(self, file_path: str) -> Dict[str, Any]:
        """
        وارد کردن دانش لهجه‌ای از یک فایل JSON.

        Returns:
            تغییرات ایجاد شده در تعداد رکوردها.
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"فایل در مسیر {file_path} یافت نشد.")
            with open(file_path, "r", encoding="utf-8") as f:
                import_data = json.load(f)
            before_stats = {
                "dialects": len(self.dialects),
                "features": len(self.dialect_features),
                "words": len(self.dialect_words),
                "rules": len(self.dialect_conversion_rules)
            }
            if "dialects" in import_data and isinstance(import_data["dialects"], list):
                for dialect in import_data["dialects"]:
                    if "dialect_id" in dialect:
                        self._store_dialect(dialect)
                        self.dialects[dialect["dialect_id"]] = dialect
            if "features" in import_data and isinstance(import_data["features"], list):
                for feature in import_data["features"]:
                    if "feature_id" in feature:
                        self._store_dialect_feature(feature)
                        self.dialect_features[feature["feature_id"]] = feature
            if "words" in import_data and isinstance(import_data["words"], list):
                for word in import_data["words"]:
                    if "word_id" in word:
                        self._store_dialect_word(word)
                        self.dialect_words[word["word_id"]] = word
            if "conversion_rules" in import_data and isinstance(import_data["conversion_rules"], list):
                for rule in import_data["conversion_rules"]:
                    if "rule_id" in rule:
                        self._store_conversion_rule(rule)
                        self.dialect_conversion_rules[rule["rule_id"]] = rule
            after_stats = {
                "dialects": len(self.dialects),
                "features": len(self.dialect_features),
                "words": len(self.dialect_words),
                "rules": len(self.dialect_conversion_rules)
            }
            changes = {
                "dialects_added": after_stats["dialects"] - before_stats["dialects"],
                "features_added": after_stats["features"] - before_stats["features"],
                "words_added": after_stats["words"] - before_stats["words"],
                "rules_added": after_stats["rules"] - before_stats["rules"],
                "import_timestamp": import_data.get("timestamp", 0),
                "import_time_formatted": self._format_time(
                    time.time() - import_data.get("timestamp", 0)) + " قبل" if "timestamp" in import_data else "نامشخص"
            }
            self.statistics["new_dialect_features_discovered"] = len(self.dialect_features)
            self.statistics["new_dialect_words_discovered"] = len(self.dialect_words)
            return changes
        except Exception as e:
            self.logger.error(f"خطا در وارد کردن دانش لهجه‌ای: {e}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """دریافت آمار کلی عملکرد سیستم."""
        uptime = time.time() - self.statistics["start_time"]
        stats = {
            "total_dialects": len(self.dialects),
            "total_features": len(self.dialect_features),
            "total_words": len(self.dialect_words),
            "total_rules": len(self.dialect_conversion_rules),
            "uptime_seconds": uptime,
            "uptime_formatted": self._format_time(uptime),
            "smart_model_available": False,  # در این لایه، مدل‌ها مدیریت نمی‌شوند
            "teacher_available": False
        }
        for key, value in self.statistics.items():
            if key != "start_time":
                stats[key] = value
        if stats["requests"] > 0:
            stats["cache_hit_rate"] = round(stats["cache_hits"] / stats["requests"] * 100, 2)
        else:
            stats["cache_hit_rate"] = 0
        stats["system_health"] = self.health_check.check_all_services()
        return stats

    def _format_time(self, seconds: float) -> str:
        """قالب‌بندی زمان به صورت خوانا."""
        days, remainder = divmod(int(seconds), 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_parts = []
        if days > 0:
            time_parts.append(f"{days} روز")
        if hours > 0:
            time_parts.append(f"{hours} ساعت")
        if minutes > 0:
            time_parts.append(f"{minutes} دقیقه")
        if seconds > 0 or not time_parts:
            time_parts.append(f"{seconds} ثانیه")
        return " و ".join(time_parts)

    def _learn_from_message(self, message: Dict[str, Any]):
        """یادگیری از پیام‌های دریافتی از Kafka."""
        try:
            if "text" in message and "data" in message:
                text = message["text"]
                data = message["data"]
                # در این لایه، فقط ثبت می‌کنیم؛ فراخوانی متدهای مدل در لایه پردازش انجام می‌شود.
                self.logger.info("Received learning message.")
        except Exception as e:
            self.logger.error(f"خطا در یادگیری از پیام: {e}")

    def _learn_from_teacher(self, text: Any, teacher_output: Any) -> bool:
        """
        در صورت امکان، داده‌های آموزشی را به مدل دانش‌آموز منتقل می‌کند.
        این تابع در این لایه فقط ثبت می‌شود.
        """
        try:
            self.logger.info("Learning from teacher output...")
            # فرض کنید فراخوانی مدل دانش‌آموز انجام می‌شود؛ در اینجا فقط نتیجه True را برمی‌گردانیم.
            return True
        except Exception as e:
            self.logger.error(f"خطا در یادگیری از معلم: {e}")
            return False

    def _publish_learning_data(self, text: Any, data: Any):
        """انتشار داده‌های آموزشی از طریق Kafka."""
        try:
            if isinstance(text, torch.Tensor):
                text_str = "tensor_data"
            else:
                text_str = text
            if isinstance(data, torch.Tensor):
                data_json = data.cpu().numpy().tolist()
            elif isinstance(data, np.ndarray):
                data_json = data.tolist()
            else:
                data_json = data
            message = {
                "type": "dialect_learning",
                "text": text_str,
                "data": data_json,
                "timestamp": time.time()
            }
            self.kafka_producer.send_message("dialect_updates", json.dumps(message))
        except Exception as e:
            self.logger.error(f"خطا در انتشار داده یادگیری: {e}")

    def get_all_dialects(self) -> List[Dict[str, Any]]:
        """دریافت لیست تمامی لهجه‌ها."""
        return list(self.dialects.values())

    def get_dialect_features(self, dialect_code: str) -> List[Dict[str, Any]]:
        """دریافت ویژگی‌های یک لهجه خاص."""
        dialect = self.get_dialect_by_code(dialect_code)
        if not dialect:
            return []
        dialect_id = dialect.get("dialect_id")
        features = [f for f in self.dialect_features.values() if f.get("dialect_id") == dialect_id]
        return sorted(features, key=lambda x: x.get("confidence", 0), reverse=True)

    def get_dialect_words(self, dialect_code: str) -> List[Dict[str, Any]]:
        """دریافت واژگان یک لهجه خاص."""
        dialect = self.get_dialect_by_code(dialect_code)
        if not dialect:
            return []
        dialect_id = dialect.get("dialect_id")
        words = [w for w in self.dialect_words.values() if w.get("dialect_id") == dialect_id]
        return sorted(words, key=lambda x: x.get("confidence", 0), reverse=True)

    def get_conversion_rules(self, source_dialect_code: str, target_dialect_code: str) -> List[Dict[str, Any]]:
        """دریافت قواعد تبدیل بین دو لهجه."""
        source = self.get_dialect_by_code(source_dialect_code)
        target = self.get_dialect_by_code(target_dialect_code)
        if not source or not target:
            return []
        source_id = source.get("dialect_id")
        target_id = target.get("dialect_id")
        rules = [r for r in self.dialect_conversion_rules.values() if
                 r.get("source_dialect") == source_id and r.get("target_dialect") == target_id]
        return sorted(rules, key=lambda x: (x.get("rule_type"), -x.get("confidence", 0)))

    def find_similar_dialects(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """یافتن لهجه‌های مشابه بر اساس تطابق ویژگی‌ها و واژگان."""
        self.statistics["requests"] += 1
        normalized_text = self._normalize_text(text)
        cache_key = f"similar_dialects:{normalized_text}:{top_k}"
        cached = self.cache_manager.get_cached_result(cache_key)
        if cached:
            self.statistics["cache_hits"] += 1
            return json.loads(cached)
        dialect_scores = defaultdict(float)
        dialect_feats = defaultdict(list)
        dialect_words = defaultdict(list)
        for feature_id, feature in self.dialect_features.items():
            dialect_id = feature.get("dialect_id")
            pattern = feature.get("feature_pattern")
            if not dialect_id or not pattern:
                continue
            try:
                matches = list(re.finditer(pattern, normalized_text, re.IGNORECASE))
                if matches:
                    weight = feature.get("confidence", 0.8) * self.detection_params.get("feature_weight", 0.7)
                    score = weight * len(matches)
                    dialect_scores[dialect_id] += score
                    dialect_feats[dialect_id].append({
                        "feature_id": feature_id,
                        "description": feature.get("description"),
                        "count": len(matches)
                    })
            except Exception as e:
                self.logger.error(f"خطا در پردازش ویژگی {feature_id}: {e}")
        for word_id, word in self.dialect_words.items():
            dialect_id = word.get("dialect_id")
            w = word.get("word")
            if not dialect_id or not w:
                continue
            try:
                pattern = r'\b' + re.escape(w) + r'\b'
                matches = list(re.finditer(pattern, normalized_text, re.IGNORECASE))
                if matches:
                    weight = word.get("confidence", 0.8) * self.detection_params.get("word_weight", 0.3)
                    score = weight * len(matches)
                    dialect_scores[dialect_id] += score
                    dialect_words[dialect_id].append({
                        "word": w,
                        "standard": word.get("standard_equivalent"),
                        "count": len(matches)
                    })
            except Exception as e:
                self.logger.error(f"خطا در پردازش واژه {word_id}: {e}")
        results = []
        for did, score in dialect_scores.items():
            if did in self.dialects:
                results.append({
                    "dialect_id": did,
                    "dialect_name": self.dialects[did].get("dialect_name", ""),
                    "dialect_code": self.dialects[did].get("dialect_code", ""),
                    "score": score,
                    "features": dialect_feats[did],
                    "words": dialect_words[did]
                })
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
        self.cache_manager.cache_result(cache_key, json.dumps(sorted_results), 86400)
        return sorted_results

    def get_dialect_vector(self, text: str, dialect_code: Optional[str] = None) -> List[float]:
        """استخراج بردار معنایی متن با لهجه مشخص."""
        self.statistics["requests"] += 1
        normalized_text = self._normalize_text(text)
        if dialect_code is None:
            detected = self.get_all_dialects().get("STANDARD")
            dialect_code = "STANDARD" if not detected else detected.get("dialect_code", "STANDARD")
        cache_key = f"dialect_vector:{normalized_text}:{dialect_code}"
        cached = self.cache_manager.get_cached_result(cache_key)
        if cached:
            self.statistics["cache_hits"] += 1
            return json.loads(cached)
        dialect = self.get_dialect_by_code(dialect_code)
        if not dialect:
            return []
        dialect_id = dialect.get("dialect_id")
        text_hash = str(hash(normalized_text))
        stored = self.database.execute_query(
            f"SELECT vector FROM dialect_text_vectors WHERE text_hash='{text_hash}' AND dialect_id='{dialect_id}' LIMIT 1"
        )
        if stored and len(stored) > 0 and "vector" in stored[0]:
            vector = stored[0]["vector"]
            self.cache_manager.cache_result(cache_key, json.dumps(vector), 86400)
            return vector
        vector = None
        if self.smart_model and self.smart_model.confidence_level(normalized_text) >= CONFIG["confidence_threshold"]:
            self.statistics["smart_model_uses"] += 1
            try:
                with torch.no_grad():
                    vector = self.smart_model.get_dialect_vector(normalized_text, dialect_code)
                    if isinstance(vector, torch.Tensor):
                        vector = vector.cpu().numpy().tolist()
            except Exception as e:
                self.logger.error(f"خطا در محاسبه بردار با مدل هوشمند: {e}")
        if vector is None and self.teacher:
            self.statistics["teacher_uses"] += 1
            try:
                with torch.no_grad():
                    vector = self.teacher.get_dialect_vector(normalized_text, dialect_code)
                    if isinstance(vector, torch.Tensor):
                        vector = vector.cpu().numpy().tolist()
                    self._learn_from_teacher(normalized_text, {"vector": vector, "dialect": dialect_code})
            except Exception as e:
                self.logger.error(f"خطا در محاسبه بردار با مدل معلم: {e}")
        if vector is None:
            vector = list(np.random.randn(128))
        try:
            self.database.insert_data("dialect_text_vectors", {
                "text_hash": text_hash,
                "text": normalized_text,
                "dialect_id": dialect_id,
                "vector": vector
            })
            self.vector_store.insert_vectors("dialect_vectors", [{
                "id": f"{text_hash}_{dialect_id}",
                "vector": vector,
                "text": normalized_text,
                "dialect_id": dialect_id
            }])
        except Exception as e:
            self.logger.error(f"خطا در ذخیره‌سازی بردار: {e}")
        self.cache_manager.cache_result(cache_key, json.dumps(vector), 86400)
        return vector

    def get_statistics(self) -> Dict[str, Any]:
        """دریافت آمار کلی سیستم."""
        uptime = time.time() - self.statistics["start_time"]
        stats = {
            "total_dialects": len(self.dialects),
            "total_features": len(self.dialect_features),
            "total_words": len(self.dialect_words),
            "total_rules": len(self.dialect_conversion_rules),
            "uptime_seconds": uptime,
            "uptime_formatted": self._format_time(uptime),
            "smart_model_available": self.smart_model is not None,
            "teacher_available": self.teacher is not None
        }
        for key, value in self.statistics.items():
            if key != "start_time":
                stats[key] = value
        if stats["requests"] > 0:
            stats["cache_hit_rate"] = round(stats["cache_hits"] / stats["requests"] * 100, 2)
        else:
            stats["cache_hit_rate"] = 0
        stats["system_health"] = self.health_check.check_all_services()
        return stats

    def _format_time(self, seconds: float) -> str:
        """قالب‌بندی زمان به صورت خوانا."""
        days, rem = divmod(int(seconds), 86400)
        hours, rem = divmod(rem, 3600)
        minutes, secs = divmod(rem, 60)
        parts = []
        if days > 0:
            parts.append(f"{days} روز")
        if hours > 0:
            parts.append(f"{hours} ساعت")
        if minutes > 0:
            parts.append(f"{minutes} دقیقه")
        if secs > 0 or not parts:
            parts.append(f"{secs} ثانیه")
        return " و ".join(parts)

    def get_dialect_by_code(self, dialect_code: str) -> Optional[Dict[str, Any]]:
        """یافتن لهجه بر اساس کد لهجه."""
        for d in self.dialects.values():
            if d.get("dialect_code") == dialect_code:
                return d
        return None
