"""
KnowledgeIntegrator Module
----------------------------
این فایل مسئول یکپارچه‌سازی دانش جدید استخراج‌شده با دانش موجود در مدل است.
این کلاس از BaseComponent ارث‌بری می‌کند و وظیفه دارد:
  - دریافت دانش جدید به صورت دیکشنری.
  - بررسی وجود موضوع (topic) در پایگاه دانش موجود.
  - در صورت وجود، ادغام دانش جدید با دانش قبلی به‌صورت هوشمند (merge و به‌روزرسانی).
  - در صورت عدم وجود، اضافه کردن دانش جدید به پایگاه دانش.
  - مدیریت تعارضات و ثبت تغییرات به صورت دقیق.
این نسخه نهایی و عملیاتی با بهترین مکانیسم‌های هوشمند و کارایی بالا پیاده‌سازی شده است.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from ..base.base_component import BaseComponent


class KnowledgeIntegrator(BaseComponent):
    """
    KnowledgeIntegrator مسئول یکپارچه‌سازی دانش جدید با پایگاه دانش موجود است.

    ویژگی‌ها:
      - نگهداری یک پایگاه دانش داخلی به صورت دیکشنری.
      - متد integrate_knowledge برای ادغام دانش جدید با دانش موجود.
      - ثبت رویدادهای تغییر دانش جهت نظارت و ثبت متریک‌ها.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(component_type="knowledge_integrator", config=config)
        self.logger = logging.getLogger("KnowledgeIntegrator")
        # پایگاه دانش به صورت دیکشنری؛ کلید: topic, مقدار: دیکشنری از اطلاعات دانش
        self.knowledge_base: Dict[str, Dict[str, Any]] = {}
        # تنظیمات ادغام دانش، مانند روش ادغام یا آستانه‌های تعارض
        self.merge_strategy = self.config.get("merge_strategy", "concatenate")
        self.logger.info(f"[KnowledgeIntegrator] Initialized with merge_strategy={self.merge_strategy}")

    def integrate_knowledge(self, new_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """
        یکپارچه‌سازی دانش جدید با پایگاه دانش موجود.

        Args:
            new_knowledge (Dict[str, Any]): دانش جدید شامل:
                - topic: نام موضوع دانش (ضروری)
                - content: محتوای دانش (متنی یا ساختار یافته)
                - source: منبع دانش (اختیاری)
                - timestamp: زمان استخراج دانش (اختیاری)
                - additional_info: سایر جزئیات (اختیاری)

        Returns:
            Dict[str, Any]: نتیجه یکپارچه‌سازی شامل:
                - status: "merged" یا "added"
                - topic: موضوع دانش
                - merged_content: محتوای نهایی دانش برای موضوع
                - details: توضیحات مربوط به روند ادغام
        """
        try:
            topic = new_knowledge.get("topic")
            if not topic:
                raise ValueError("New knowledge must include a 'topic' field.")

            new_content = new_knowledge.get("content", "")
            new_source = new_knowledge.get("source", "unknown")
            new_timestamp = new_knowledge.get("timestamp", datetime.utcnow().isoformat())
            additional_info = new_knowledge.get("additional_info", {})

            # اگر موضوع دانش از قبل وجود داشته باشد، ادغام را انجام می‌دهیم
            if topic in self.knowledge_base:
                existing_entry = self.knowledge_base[topic]
                merged_content = self._merge_contents(existing_entry.get("content", ""), new_content)
                # به‌روزرسانی پایگاه دانش
                self.knowledge_base[topic] = {
                    "content": merged_content,
                    "source": f"{existing_entry.get('source')} | {new_source}",
                    "last_updated": new_timestamp,
                    "additional_info": {**existing_entry.get("additional_info", {}), **additional_info}
                }
                status = "merged"
                details = "Existing knowledge merged with new content."
            else:
                # اضافه کردن دانش جدید به پایگاه دانش
                self.knowledge_base[topic] = {
                    "content": new_content,
                    "source": new_source,
                    "last_updated": new_timestamp,
                    "additional_info": additional_info
                }
                merged_content = new_content
                status = "added"
                details = "New knowledge added to the knowledge base."

            # ثبت رویداد تغییر دانش (در صورت نیاز می‌توان trigger_event از BaseComponent فراخوانی کرد)
            self.logger.info(f"[KnowledgeIntegrator] Knowledge for topic '{topic}' {status}.")
            self.increment_metric(f"knowledge_{status}")
            return {
                "status": status,
                "topic": topic,
                "merged_content": merged_content,
                "details": details
            }
        except Exception as e:
            self.logger.error(f"[KnowledgeIntegrator] Error integrating knowledge: {str(e)}")
            self.record_error_metric()
            return {"status": "error", "error": str(e)}

    def _merge_contents(self, existing: Any, new: Any) -> Any:
        """
        ادغام محتوای دانش. در این نسخه نهایی، استراتژی ادغام به صورت "concatenate" تنظیم شده است.
        در صورت نیاز می‌توان استراتژی‌های پیشرفته‌تری مانند ترکیب معنایی یا انتخاب محتوا را پیاده‌سازی کرد.

        Args:
            existing (Any): محتوای دانش موجود.
            new (Any): محتوای دانش جدید.

        Returns:
            Any: محتوای ادغام‌شده.
        """
        if self.merge_strategy == "concatenate":
            # اگر محتوا رشته‌ای باشد، با یک فضای خالی ادغام می‌شود.
            if isinstance(existing, str) and isinstance(new, str):
                if existing and new:
                    return existing + " " + new
                else:
                    return existing or new
            # در غیر این صورت، به صورت لیستی ترکیب می‌شود.
            else:
                combined = []
                if isinstance(existing, list):
                    combined.extend(existing)
                else:
                    combined.append(existing)
                if isinstance(new, list):
                    combined.extend(new)
                else:
                    combined.append(new)
                return combined
        else:
            # سایر استراتژی‌ها می‌توانند در اینجا اضافه شوند.
            return new

    def get_knowledge(self, topic: Optional[str] = None) -> Dict[str, Any]:
        """
        دریافت دانش از پایگاه.

        Args:
            topic (Optional[str]): اگر مشخص شود، دانش مربوط به آن موضوع برگردانده می‌شود؛ در غیر این صورت، کل پایگاه دانش.

        Returns:
            Dict[str, Any]: دانش موجود.
        """
        if topic:
            return self.knowledge_base.get(topic, {})
        return self.knowledge_base


# Sample usage (for testing; final version is intended for production):
if __name__ == "__main__":
    import json
    import logging

    logging.basicConfig(level=logging.DEBUG)

    ki = KnowledgeIntegrator(config={"merge_strategy": "concatenate"})

    # دانش جدید
    new_knowledge_1 = {
        "topic": "Quantum Computing",
        "content": "Quantum computing uses quantum bits for processing.",
        "source": "Research Paper",
        "additional_info": {"reliability": "high"}
    }
    result1 = ki.integrate_knowledge(new_knowledge_1)
    print(json.dumps(result1, indent=2, ensure_ascii=False))

    # دانش جدید برای همان موضوع (ادغام)
    new_knowledge_2 = {
        "topic": "Quantum Computing",
        "content": "It promises exponential speedup for certain algorithms.",
        "source": "Tech Blog",
        "additional_info": {"reliability": "medium"}
    }
    result2 = ki.integrate_knowledge(new_knowledge_2)
    print(json.dumps(result2, indent=2, ensure_ascii=False))

    # دریافت دانش نهایی برای موضوع
    final_knowledge = ki.get_knowledge("Quantum Computing")
    print("Final integrated knowledge:")
    print(json.dumps(final_knowledge, indent=2, ensure_ascii=False))
