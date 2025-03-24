"""
ماژول semantic_services.py

این ماژول منطق تجاری تحلیل معنایی متون فارسی را پیاده‌سازی می‌کند. این شامل تحلیل جامع متون، استخراج بردار معنایی،
یافتن متون مشابه و استخراج ویژگی‌های معنایی می‌باشد. این پیاده‌سازی از معماری معلم-دانش‌آموز برای شروع استفاده می‌کند
و به مرور زمان مدل دانش‌آموز (smart model) مستقل می‌شود. این فایل از سیستم‌های پیام‌رسان (Kafka) استفاده نمی‌کند و
به جای آن، از سرویس‌های کش (Redis) و پایگاه داده (ClickHouse) برای ذخیره و بازیابی نتایج بهره می‌برد.
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ai.models.language.infrastructure.caching.cache_manager import CacheManager
from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseAdapter
from ai.models.language.infrastructure.vector_store.milvus_adapter import MilvusAdapter
from ai.models.language.infrastructure.vector_store.vector_search import VectorSearch
from ai.models.language.infrastructure.monitoring.performance_metrics import PerformanceMetrics
from ai.models.language.infrastructure.monitoring.health_check import HealthCheck
from ..utils.text_normalization import TextNormalizer
from ..utils.tokenization import Tokenizer
from ..semantics.semantic_models import SemanticAnalysisResult

from ai.models.language.adaptors.persian.config import CONFIG  # تنظیمات سیستم

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SemanticServices:
    """
    کلاس SemanticServices

    این کلاس مسئول تحلیل جامع معنایی متون فارسی است. روش کار به این صورت است که متن ورودی نرمال‌سازی شده،
    سپس با استفاده از مدل هوشمند (و یا مدل معلم در صورت نیاز) تحلیل می‌شود. نتیجه تحلیل شامل هدف (intent)،
    احساسات (sentiment)، موضوعات (topics) و بردار معنایی (embedding) می‌باشد. همچنین عملکرد از طریق کش و پایگاه داده ذخیره می‌شود.
    """

    def __init__(self, language: str = "persian"):
        self.language = language
        self.logger = logger

        # ابزارهای پردازش زبان
        self.normalizer = TextNormalizer()
        self.tokenizer = Tokenizer()

        # بارگذاری مدل‌های هوشمند و معلم
        self.smart_model = self._load_smart_model()
        self.teacher = self._load_teacher_model()

        # مدل Sentence Transformer به عنوان fallback
        self.sentence_transformer = self._load_sentence_transformer()

        # زیرساخت‌های پایگاه داده و کش
        self.database = ClickHouseAdapter()  # استفاده از آداپتور همگام
        self.cache_manager = CacheManager()
        self.vector_store = MilvusAdapter(collection_name="semantic_vectors")
        self.vector_search = VectorSearch()
        self.performance_metrics = PerformanceMetrics()
        self.health_check = HealthCheck()

        # آمار عملکرد
        self.statistics = {
            "requests": 0,
            "cache_hits": 0,
            "smart_model_uses": 0,
            "teacher_uses": 0,
            "new_topics_discovered": 0,
            "start_time": time.time()
        }

    def _load_smart_model(self) -> Optional[Any]:
        try:
            import importlib
            module = importlib.import_module(f"ai.models.language.adaptors.{self.language}.smart_model")
            return module.SmartModel()
        except Exception as e:
            self.logger.error(f"Error loading smart model: {e}")
            return None

    def _load_teacher_model(self) -> Optional[Any]:
        try:
            import importlib
            module = importlib.import_module(f"ai.models.language.adaptors.{self.language}.teacher")
            return module.TeacherModel()
        except Exception as e:
            self.logger.warning(f"Teacher model not found; proceeding without it: {e}")
            return None

    def _load_sentence_transformer(self) -> Optional[SentenceTransformer]:
        try:
            import os
            model_path = "models/sentence-transformer-persian" if os.path.exists(
                "models/sentence-transformer-persian") else "paraphrase-multilingual-MiniLM-L12-v2"
            return SentenceTransformer(model_path)
        except Exception as e:
            self.logger.error(f"Error loading Sentence Transformer: {e}")
            return None

    def analyze_text(self, text: str) -> SemanticAnalysisResult:
        """
        تحلیل جامع معنایی متن شامل تشخیص هدف، احساسات، موضوعات و بردار معنایی.

        Args:
            text (str): متن ورودی

        Returns:
            SemanticAnalysisResult: نتیجه تحلیل معنایی به صورت یک مدل داده‌ای
        """
        self.statistics["requests"] += 1
        normalized_text = self.normalizer.normalize(text)
        cache_key = f"semantics:{normalized_text}"
        cached = self.cache_manager.get_cached_result(cache_key)
        if cached:
            self.statistics["cache_hits"] += 1
            result_dict = json.loads(cached)
            return SemanticAnalysisResult(**result_dict)

        # تعیین سطح اطمینان با استفاده از مدل هوشمند (در صورت وجود)
        confidence = 0.0
        source = "rule_based"
        result_data: Dict[str, Any] = {}

        if self.smart_model:
            try:
                confidence = self.smart_model.confidence_level(normalized_text)
            except Exception as e:
                self.logger.error(f"Error obtaining confidence from smart model: {e}")

        if self.smart_model and confidence >= CONFIG.get("confidence_threshold", 0.7):
            self.statistics["smart_model_uses"] += 1
            try:
                result_data = self.smart_model.analyze_semantics(normalized_text)
                source = "smart_model"
            except Exception as e:
                self.logger.error(f"Error during smart model semantic analysis: {e}")
        elif self.teacher:
            self.statistics["teacher_uses"] += 1
            try:
                result_data = self.teacher.analyze_semantics(normalized_text)
                source = "teacher"
            except Exception as e:
                self.logger.error(f"Error during teacher model semantic analysis: {e}")
        else:
            # تحلیل ترکیبی مبتنی بر قوانین ساده
            intent = self._rule_based_intent(normalized_text)
            sentiment = self._rule_based_sentiment(normalized_text)
            topics = self._rule_based_topics(normalized_text)
            vector = self.get_semantic_vector(normalized_text)
            result_data = {
                "intent": intent,
                "sentiment": sentiment,
                "topics": topics,
                "embedding": vector
            }
            source = "rule_based"
            confidence = 0.5

        # بررسی و افزودن موضوع جدید در صورت شناسایی الگوی عمومی
        if "topics" in result_data and result_data["topics"]:
            new_topic = self._discover_new_topic(normalized_text, result_data["topics"])
            if new_topic and new_topic not in result_data["topics"]:
                result_data["topics"].append(new_topic)

        self._store_semantic_result(normalized_text, result_data, confidence, source)
        self.cache_manager.cache_result(cache_key, json.dumps(result_data), ttl=86400)
        self.performance_metrics.collect_metrics({
            "semantic_analysis": {
                "text_length": len(normalized_text),
                "confidence": confidence,
                "source": source
            }
        })

        return SemanticAnalysisResult(
            text=text,
            normalized_text=normalized_text,
            intent=result_data.get("intent", ""),
            sentiment=result_data.get("sentiment", ""),
            topics=result_data.get("topics", []),
            embedding=result_data.get("embedding", []),
            confidence=confidence,
            source=source,
            timestamp=time.time()
        )

    def _rule_based_intent(self, text: str) -> str:
        lower_text = text.lower()
        if any(greet in lower_text for greet in ["سلام", "درود", "صبح بخیر", "شب بخیر"]):
            return "greeting"
        elif any(farewell in lower_text for farewell in ["خداحافظ", "بدرود", "به امید دیدار"]):
            return "farewell"
        elif "؟" in text or any(q_word in lower_text for q_word in ["چه", "چرا", "کی", "چگونه"]):
            return "question"
        elif any(cmd in lower_text for cmd in ["لطفاً", "کن", "بگو", "برو"]):
            return "command"
        else:
            return "statement"

    def _rule_based_sentiment(self, text: str) -> str:
        positive_words = ["خوب", "عالی", "زیبا", "عشق", "شاد", "خوشحال"]
        negative_words = ["بد", "ضعیف", "زشت", "نفرت", "ناراحت", "غمگین"]
        pos_count = sum(text.lower().count(word) for word in positive_words)
        neg_count = sum(text.lower().count(word) for word in negative_words)
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"

    def _rule_based_topics(self, text: str) -> List[str]:
        default_topics = {
            "سیاسی": ["دولت", "مجلس", "انتخابات"],
            "اقتصادی": ["اقتصاد", "بازار", "سرمایه"],
            "اجتماعی": ["جامعه", "خانواده"],
            "فرهنگی": ["هنر", "فرهنگ"],
            "علمی": ["دانش", "تحقیق"],
            "ورزشی": ["ورزش", "تیم"],
            "فناوری": ["فناوری", "کامپیوتر"],
            "پزشکی": ["پزشکی", "درمان"]
        }
        found = []
        lower_text = text.lower()
        for topic, keywords in default_topics.items():
            if any(keyword.lower() in lower_text for keyword in keywords):
                found.append(topic)
        return found if found else ["عمومی"]

    def _discover_new_topic(self, text: str, current_topics: List[str]) -> Optional[str]:
        if len(text) < 100:
            return None
        if current_topics == ["عمومی"]:
            if self.sentence_transformer:
                vector = self.sentence_transformer.encode(text)
                # شرط نمونه برای پیشنهاد موضوع جدید
                if vector.mean() > 0:
                    return "موضوع_جدید"
        return None

    def _store_semantic_result(self, text: str, result: Dict[str, Any], confidence: float, source: str):
        try:
            text_hash = str(hash(text))
            data = {
                "text_hash": text_hash,
                "text": text,
                "intent": result.get("intent", ""),
                "sentiment": result.get("sentiment", ""),
                "topics": result.get("topics", []),
                "confidence": float(confidence),
                "source": source,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }
            self.database.insert_data("semantic_analysis", data)
            if "embedding" in result and isinstance(result["embedding"], list):
                vector = result["embedding"]
                self.database.insert_data("semantic_vectors", {
                    "text_hash": text_hash,
                    "text": text,
                    "vector": vector,
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
                })
                self.vector_store.insert_vectors("semantic_vectors",
                                                 [{"id": text_hash, "vector": vector, "text": text}])
        except Exception as e:
            self.logger.error(f"Error storing semantic result: {e}")

    def get_semantic_vector(self, text: str) -> List[float]:
        normalized_text = self.normalizer.normalize(text)
        cache_key = f"vector:{normalized_text}"
        cached = self.cache_manager.get_cached_result(cache_key)
        if cached:
            self.statistics["cache_hits"] += 1
            return json.loads(cached)
        text_hash = str(hash(normalized_text))
        stored = self.database.execute_query(
            f"SELECT vector FROM semantic_vectors WHERE text_hash='{text_hash}' LIMIT 1")
        if stored and len(stored) > 0 and "vector" in stored[0]:
            vector = stored[0]["vector"]
            self.cache_manager.cache_result(cache_key, json.dumps(vector), ttl=86400)
            return vector
        vector: Optional[List[float]] = None
        confidence = 0.0
        source = "smart_model"
        if self.smart_model:
            try:
                confidence = self.smart_model.confidence_level(normalized_text)
                if confidence >= CONFIG.get("confidence_threshold", 0.7):
                    with torch.no_grad():
                        vector = self.smart_model.forward(normalized_text)
                        if isinstance(vector, torch.Tensor):
                            vector = vector.cpu().numpy().tolist()
            except Exception as e:
                self.logger.error(f"Error in smart model vector extraction: {e}")
        if vector is None and self.teacher:
            try:
                with torch.no_grad():
                    vector = self.teacher.forward(normalized_text)
                    if isinstance(vector, torch.Tensor):
                        vector = vector.cpu().numpy().tolist()
                self._learn_from_teacher(normalized_text, vector)
                source = "teacher"
            except Exception as e:
                self.logger.error(f"Error in teacher model vector extraction: {e}")
        if vector is None and self.sentence_transformer:
            try:
                vector = self.sentence_transformer.encode(normalized_text).tolist()
                source = "sentence_transformer"
            except Exception as e:
                self.logger.error(f"Error in sentence transformer vector extraction: {e}")
        if vector is None:
            vector = list(np.random.randn(768))
            source = "random"
        try:
            self.database.insert_data("semantic_vectors", {
                "text_hash": text_hash,
                "text": normalized_text,
                "vector": vector,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            })
            self.vector_store.insert_vectors("semantic_vectors",
                                             [{"id": text_hash, "vector": vector, "text": normalized_text}])
        except Exception as e:
            self.logger.error(f"Error storing semantic vector: {e}")
        self.cache_manager.cache_result(cache_key, json.dumps(vector), ttl=86400)
        self.performance_metrics.collect_metrics({
            "semantic_vector": {
                "text_length": len(normalized_text),
                "vector_source": source
            }
        })
        return vector

    def find_similar_texts(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        normalized_text = self.normalizer.normalize(query)
        query_vector = self.get_semantic_vector(normalized_text)
        try:
            results = self.vector_search.search_vectors("semantic_vectors", query_vector, top_k)
            enhanced_results = []
            for item in results:
                if 'text' in item:
                    text_hash = str(hash(item['text']))
                    analysis = self.database.execute_query(
                        f"SELECT intent, sentiment, topics FROM semantic_analysis WHERE text_hash='{text_hash}' LIMIT 1"
                    )
                    if analysis and len(analysis) > 0:
                        item.update(analysis[0])
                    enhanced_results.append(item)
            return enhanced_results if enhanced_results else results
        except Exception as e:
            self.logger.error(f"Error finding similar texts: {e}")
            return []

    def _learn_from_teacher(self, text: Union[str, torch.Tensor], teacher_output: Any) -> bool:
        try:
            if self.smart_model:
                learning_result = self.smart_model.learn_from_teacher(text, teacher_output)
                return learning_result
            return False
        except Exception as e:
            self.logger.error(f"Error in learning from teacher: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        total_time = time.time() - self.statistics["start_time"]
        total_requests = self.statistics["requests"]
        cache_hit_ratio = (self.statistics["cache_hits"] / total_requests) if total_requests > 0 else 0
        teacher_ratio = (self.statistics["teacher_uses"] / total_requests) if total_requests > 0 else 0
        smart_ratio = (self.statistics["smart_model_uses"] / total_requests) if total_requests > 0 else 0
        perf = self.performance_metrics.get_system_health_metrics()
        return {
            "total_requests": total_requests,
            "cache_hits": self.statistics["cache_hits"],
            "cache_hit_ratio": cache_hit_ratio,
            "teacher_uses": self.statistics["teacher_uses"],
            "teacher_ratio": teacher_ratio,
            "smart_model_uses": self.statistics["smart_model_uses"],
            "smart_model_ratio": smart_ratio,
            "new_topics_discovered": self.statistics["new_topics_discovered"],
            "uptime_seconds": total_time,
            "requests_per_second": total_requests / total_time if total_time > 0 else 0,
            "performance_metrics": perf
        }

    def clear_cache(self) -> bool:
        try:
            self.cache_manager.flush_cache()
            return True
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return False
