"""
KnowledgeCoverage Module
--------------------------
این فایل مسئول سنجش پوشش دانشی مدل در فرآیند خودآموزی است.
این کلاس میزان پوشش حوزه‌های دانشی مدل را با مقایسه دانش موجود (knowledge base)
با حوزه‌های مورد انتظار یا مجموعه‌ای از موضوعات کلیدی محاسبه می‌کند.
همچنین درصد پوشش و فواصل موجود را ارزیابی و گزارش می‌دهد.
این نسخه نهایی و عملیاتی با بهترین مکانیسم‌های هوشمند و کارایی بالا پیاده‌سازی شده است.
"""

import logging
from typing import Dict, Any, List, Optional

from ..base.base_component import BaseComponent


class KnowledgeCoverage(BaseComponent):
    """
    KnowledgeCoverage مسئول سنجش میزان پوشش دانشی مدل است.

    ورودی مورد انتظار:
      {
         "expected_topics": List[str],     # لیست حوزه‌ها یا موضوعات مورد انتظار
         "knowledge_base": Dict[str, Any]    # پایگاه دانش موجود به صورت دیکشنری که کلیدها موضوع و مقدارها محتوا یا امتیاز پوشش است.
      }

    خروجی:
      {
         "coverage_ratio": float,           # نسبت پوشش (بین 0 تا 1)
         "missing_topics": List[str],         # لیست موضوعاتی که پوشش داده نشده‌اند
         "details": str                     # توضیحات تکمیلی
      }
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(component_type="knowledge_coverage", config=config)
        self.logger = logging.getLogger("KnowledgeCoverage")
        # آستانه پوشش پیش‌فرض (مثلاً 0.8 به معنی 80 درصد پوشش)
        self.default_threshold = float(self.config.get("coverage_threshold", 0.8))
        self.logger.info(f"[KnowledgeCoverage] Initialized with default_threshold={self.default_threshold}")

    def evaluate_coverage(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        ارزیابی پوشش دانشی مدل بر اساس موضوعات مورد انتظار و دانش موجود.

        Args:
            input_data (Dict[str, Any]): شامل:
                - expected_topics: لیستی از موضوعات کلیدی مورد انتظار.
                - knowledge_base: دیکشنری از دانش موجود؛ کلیدها موضوع‌ها و مقادیر می‌توانند امتیاز یا تعداد دفعات پوشش را نشان دهند.

        Returns:
            Dict[str, Any]: شامل نسبت پوشش، موضوعات مفقود و توضیحات.
        """
        expected_topics: List[str] = input_data.get("expected_topics", [])
        knowledge_base: Dict[str, Any] = input_data.get("knowledge_base", {})

        if not expected_topics:
            self.logger.warning("[KnowledgeCoverage] No expected topics provided.")
            return {"coverage_ratio": 0.0, "missing_topics": [], "details": "No expected topics provided."}

        # محاسبه پوشش: تعداد موضوعاتی که در knowledge_base موجود هستند
        covered_topics = [topic for topic in expected_topics if topic in knowledge_base and knowledge_base[topic]]
        coverage_ratio = len(covered_topics) / len(expected_topics)
        missing_topics = [topic for topic in expected_topics if topic not in covered_topics]

        details = f"Coverage ratio is {coverage_ratio:.2f}. Missing topics: {missing_topics}."
        self.logger.info(f"[KnowledgeCoverage] Evaluation result: {details}")
        self.increment_metric("knowledge_coverage_evaluated")
        return {
            "coverage_ratio": round(coverage_ratio, 3),
            "missing_topics": missing_topics,
            "details": details
        }


# Sample usage for testing (final version intended for production)
if __name__ == "__main__":
    import logging
    import json

    logging.basicConfig(level=logging.DEBUG)

    # نمونه داده‌های ورودی
    input_data = {
        "expected_topics": ["machine learning", "deep learning", "NLP", "computer vision", "reinforcement learning"],
        "knowledge_base": {
            "machine learning": {"content": "Basic ML techniques", "score": 0.9},
            "NLP": {"content": "Natural language processing methods", "score": 0.8},
            "computer vision": {"content": "Image processing and CV", "score": 0.7}
        }
    }

    kc = KnowledgeCoverage(config={"coverage_threshold": 0.8})
    result = kc.evaluate_coverage(input_data)
    print("Knowledge Coverage Evaluation:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
