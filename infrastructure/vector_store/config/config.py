# infrastructure/vector_store/config/config.py
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Any
from dotenv import load_dotenv

# بارگذاری متغیرهای محیطی از .env
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class MilvusConfig:
    """
    کلاس مرکزی مدیریت تنظیمات Milvus - ادغام تمامی تنظیمات مرتبط با Milvus
    """
    # ------ تنظیمات اتصال پایه ------
    host: str = os.getenv("MILVUS_HOST", "localhost")
    port: int = int(os.getenv("MILVUS_PORT", "19530"))
    user: str = os.getenv("MILVUS_USER", "root")
    password: str = os.getenv("MILVUS_PASSWORD", "milvus")

    # ------ تنظیمات کالکشن ------
    default_collection: str = os.getenv("MILVUS_DEFAULT_COLLECTION", "vector_data")
    vector_dimensions: int = int(os.getenv("MILVUS_VECTOR_DIMENSIONS", "512"))
    distance_metric: str = os.getenv("MILVUS_DISTANCE_METRIC", "L2")

    # ------ تنظیمات شاردینگ و تکرار ------
    collection_shards: int = int(os.getenv("MILVUS_COLLECTION_SHARDS", "2"))
    replica_factor: int = int(os.getenv("MILVUS_REPLICA_FACTOR", "1"))

    # ------ تنظیمات ارتباطی ------
    timeout: int = int(os.getenv("MILVUS_TIMEOUT", "30"))
    max_connections: int = int(os.getenv("MILVUS_MAX_CONNECTIONS", "10"))
    retry_attempts: int = int(os.getenv("MILVUS_RETRY_ATTEMPTS", "3"))
    retry_delay: float = float(os.getenv("MILVUS_RETRY_DELAY", "1.0"))

    # ------ تنظیمات ایندکس ------
    index_type: str = os.getenv("MILVUS_INDEX_TYPE", "IVF_FLAT")
    ivf_nlist: int = int(os.getenv("MILVUS_IVF_NLIST", "1024"))
    hnsw_m: int = int(os.getenv("MILVUS_HNSW_M", "16"))
    hnsw_ef: int = int(os.getenv("MILVUS_HNSW_EF", "200"))
    index_build_threads: int = int(os.getenv("MILVUS_INDEX_BUILD_THREADS", "4"))

    def __post_init__(self):
        """
        اعتبارسنجی تنظیمات بعد از ایجاد شی
        """
        self._validate_settings()
        logger.info("Milvus configuration loaded successfully.")

    def _validate_settings(self):
        """
        بررسی اعتبار تنظیمات
        """
        valid_index_types = ["IVF_FLAT", "IVF_SQ8", "HNSW", "FLAT"]
        if self.index_type not in valid_index_types:
            logger.warning(f"Invalid MILVUS_INDEX_TYPE: {self.index_type}, falling back to 'IVF_FLAT'")
            self.index_type = "IVF_FLAT"

        valid_metrics = ["L2", "IP", "COSINE"]
        if self.distance_metric not in valid_metrics:
            logger.warning(f"Invalid MILVUS_DISTANCE_METRIC: {self.distance_metric}, falling back to 'L2'")
            self.distance_metric = "L2"

        # حداقل یک اتصال باید وجود داشته باشد
        if self.max_connections < 1:
            logger.warning(f"Invalid MILVUS_MAX_CONNECTIONS: {self.max_connections}, should be at least 1")
            self.max_connections = 1

    def get_connection_params(self) -> Dict[str, Any]:
        """
        پارامترهای اتصال به Milvus را برمی‌گرداند

        Returns:
            Dict[str, Any]: دیکشنری پارامترهای اتصال
        """
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "timeout": self.timeout
        }

    def get_collection_config(self) -> Dict[str, Any]:
        """
        تنظیمات کالکشن را برمی‌گرداند

        Returns:
            Dict[str, Any]: دیکشنری تنظیمات کالکشن
        """
        return {
            "collection_name": self.default_collection,
            "vector_dimensions": self.vector_dimensions,
            "distance_metric": self.distance_metric,
            "collection_shards": self.collection_shards,
            "replica_factor": self.replica_factor
        }

    def get_index_config(self) -> Dict[str, Any]:
        """
        تنظیمات ایندکس را برمی‌گرداند

        Returns:
            Dict[str, Any]: دیکشنری تنظیمات ایندکس
        """
        return {
            "index_type": self.index_type,
            "ivf_nlist": self.ivf_nlist,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef": self.hnsw_ef,
            "index_build_threads": self.index_build_threads
        }

    def as_dict(self) -> Dict[str, Any]:
        """
        تمام تنظیمات را به صورت دیکشنری برمی‌گرداند

        Returns:
            Dict[str, Any]: دیکشنری تمام تنظیمات
        """
        return {
            # تنظیمات اتصال
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,

            # تنظیمات کالکشن
            "default_collection": self.default_collection,
            "vector_dimensions": self.vector_dimensions,
            "distance_metric": self.distance_metric,
            "collection_shards": self.collection_shards,
            "replica_factor": self.replica_factor,

            # تنظیمات ارتباطی
            "timeout": self.timeout,
            "max_connections": self.max_connections,
            "retry_attempts": self.retry_attempts,
            "retry_delay": self.retry_delay,

            # تنظیمات ایندکس
            "index_type": self.index_type,
            "ivf_nlist": self.ivf_nlist,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef": self.hnsw_ef,
            "index_build_threads": self.index_build_threads
        }


# ایجاد نمونه پیش‌فرض قابل استفاده در سراسر کد
config = MilvusConfig()
