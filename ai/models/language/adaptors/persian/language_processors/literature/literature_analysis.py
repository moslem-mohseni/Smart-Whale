# persian/language_processors/literature/literature_analysis.py
"""
ماژول literature_analysis.py

این فایل منطق اصلی تحلیل متون ادبی فارسی را پیاده‌سازی می‌کند.
امکانات اصلی:
  - تشخیص اینکه متن شعر است یا نثر (_is_poetry)
  - تحلیل سبک ادبی (analyze_style) با استفاده از ترکیب روش‌های rule‑based و مدل‌های هوشمند (در صورت وجود)
  - تحلیل وزن شعر (analyze_meter) برای متون شعری
  - تشخیص آرایه‌های ادبی (detect_devices) با استفاده از الگوهای regex
  - شناسایی قالب شعری (identify_poetry_form) بر مبنای قافیه و ساختار
  - ارائه خروجی یکپارچه به‌صورت دیکشنری
"""

import logging
import re
import time
from difflib import SequenceMatcher
from typing import Dict, Any, List, Optional

from ..utils.text_normalization import TextNormalizer
from ..utils.tokenization import Tokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LiteratureAnalysis:
    def __init__(self,
                 smart_model: Optional[Any] = None,
                 teacher_model: Optional[Any] = None,
                 literary_styles: Optional[Dict[str, Dict[str, Any]]] = None,
                 literary_periods: Optional[Dict[str, Dict[str, Any]]] = None,
                 poetry_meters: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        سازنده LiteratureAnalysis

        Args:
            smart_model: مدل هوشمند برای تحلیل (اختیاری)
            teacher_model: مدل معلم برای تحلیل (اختیاری)
            literary_styles: دیکشنری سبک‌های ادبی (از داده‌های literature_data)
            literary_periods: دیکشنری دوره‌های ادبی
            poetry_meters: دیکشنری وزن‌های شعری
        """
        self.smart_model = smart_model
        self.teacher_model = teacher_model
        self.literary_styles = literary_styles if literary_styles is not None else {}
        self.literary_periods = literary_periods if literary_periods is not None else {}
        self.poetry_meters = poetry_meters if poetry_meters is not None else {}
        self.normalizer = TextNormalizer()
        self.tokenizer = Tokenizer()

    def _is_poetry(self, text: str) -> bool:
        """
        تشخیص اولیه اینکه متن شعر است یا نثر با استفاده از معیارهای ساده.

        Args:
            text: متن ورودی (نرمال‌شده)

        Returns:
            True اگر متن احتمالاً شعر است، False در غیر این صورت.
        """
        lines = text.strip().split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        if len(non_empty_lines) < 2:
            return False
        avg_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
        return avg_length < 50  # معیار تقریبی

    def analyze_style(self, text: str) -> Dict[str, Any]:
        """
        تحلیل سبک ادبی متن با استفاده از روش‌های rule-based (با الگوهای regex) و در صورت امکان،
        استفاده از مدل‌های هوشمند.

        Args:
            text: متن ورودی (نرمال‌شده)

        Returns:
            دیکشنری شامل اطلاعات سبک (style_id, style_name, confidence, features, source)
        """
        result = {}
        source = "rule_based"
        try:
            if self.smart_model and hasattr(self.smart_model, "detect_literary_style"):
                result = self.smart_model.detect_literary_style(text)
                if result and result.get("confidence", 0) >= 0.6:
                    source = "smart_model"
        except Exception as e:
            logger.error(f"Error in smart model style detection: {e}")

        if not result:
            style_patterns = {
                "s_1001": {"name": "سبک خراسانی", "pattern": r"\b(فعولن فعولن فعولن فعل)\b", "weight": 1.0},
                "s_1002": {"name": "سبک عراقی", "pattern": r"\b(عشق|عاشق|دلتنگ)\b", "weight": 1.0},
                "s_1003": {"name": "سبک هندی", "pattern": r"\b(خیال|مضمون|تصویر)\b", "weight": 1.0},
                "s_1004": {"name": "سبک رمانتیک", "pattern": r"\b(تنهایی|غم|اندوه)\b", "weight": 1.0},
                "s_1005": {"name": "نوگرایی (مدرنیسم)", "pattern": r"\b(شعر نو|آزاد|سپید)\b", "weight": 1.0}
            }
            scores = {}
            features = {}
            for style_id, info in style_patterns.items():
                matches = re.findall(info["pattern"], text, flags=re.IGNORECASE)
                score = len(matches) * info["weight"]
                scores[style_id] = score
                if matches:
                    features[style_id] = {
                        "pattern": info["pattern"],
                        "matches": matches[:3],
                        "count": len(matches)
                    }
            if scores:
                best_style = max(scores, key=lambda k: scores[k])
                confidence = min(scores[best_style] / 5.0, 0.95)
                result = {
                    "style_id": best_style,
                    "style_name": style_patterns[best_style]["name"],
                    "confidence": confidence,
                    "features": features.get(best_style, [])
                }
            else:
                result = {
                    "style_id": "s_1005",
                    "style_name": "نوگرایی (مدرنیسم)",
                    "confidence": 0.5,
                    "features": []
                }
            source = "rule_based"
        result["source"] = source
        return result

    def analyze_meter(self, text: str) -> Dict[str, Any]:
        """
        تحلیل وزن شعر برای متون شعری با استفاده از روش‌های rule-based.

        Args:
            text: متن ورودی (نرمال‌شده)

        Returns:
            دیکشنری شامل اطلاعات وزن (meter_id, meter_name, confidence, examples, message)
        """
        meter_scores = {}
        for meter_id, meter in self.poetry_meters.items():
            meter_pattern = meter.get("meter_pattern", "")
            if meter_pattern and re.search(re.escape(meter_pattern), text):
                meter_scores[meter_id] = 5

        if meter_scores:
            best_meter = max(meter_scores, key=lambda k: meter_scores[k])
            confidence = min(meter_scores[best_meter] / 5.0, 0.95)
            return {
                "meter_id": best_meter,
                "meter_name": self.poetry_meters.get(best_meter, {}).get("meter_name", "نامشخص"),
                "confidence": confidence,
                "examples": self.poetry_meters.get(best_meter, {}).get("examples", [])[:2],
                "message": "وزن تشخیص داده شد"
            }
        else:
            return {
                "meter_id": "",
                "meter_name": "شعر نو/سپید",
                "confidence": 0.6,
                "examples": [],
                "message": "وزن عروضی کلاسیک شناسایی نشد"
            }

    def detect_devices(self, text: str) -> List[Dict[str, Any]]:
        """
        تشخیص آرایه‌های ادبی در متن ادبی با استفاده از الگوهای regex.

        Args:
            text: متن ورودی (نرمال‌شده)

        Returns:
            لیست دیکشنری‌هایی شامل device_id، device_name، confidence و نمونه‌های تطابق.
        """
        devices_found = []
        tashbih_pattern = r'(\S+)\s+(چون|مثل|مانند|همچو|به سان)\s+(\S+)'
        matches = re.findall(tashbih_pattern, text, flags=re.IGNORECASE)
        if matches:
            devices_found.append({
                "device_id": "d_tashbih",
                "device_name": "تشبیه",
                "confidence": 0.85,
                "matches": matches[:3],
                "count": len(matches)
            })
        words = re.findall(r'\b\w+\b', text)
        jenas_pairs = []
        for i in range(len(words)):
            for j in range(i + 1, min(i + 6, len(words))):
                sim = SequenceMatcher(None, words[i], words[j]).ratio()
                if 0.6 < sim < 0.95:
                    jenas_pairs.append((words[i], words[j]))
        if jenas_pairs:
            devices_found.append({
                "device_id": "d_jenas",
                "device_name": "جناس",
                "confidence": 0.8,
                "matches": jenas_pairs[:3],
                "count": len(jenas_pairs)
            })
        return devices_found

    def identify_poetry_form(self, text: str, meter_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        تشخیص قالب شعری بر مبنای تحلیل قافیه و ساختار ابیات.

        Args:
            text: متن شعر (نرمال‌شده)
            meter_info: نتایج تحلیل وزن

        Returns:
            دیکشنری شامل form_name، form_code و confidence.
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if len(lines) == 4:
            return {
                "form_name": "رباعی",
                "form_code": "ROBAI",
                "confidence": 0.9
            }
        elif len(lines) >= 6:
            return {
                "form_name": "مثنوی",
                "form_code": "MASNAVI",
                "confidence": 0.8
            }
        else:
            return {
                "form_name": "نامشخص",
                "form_code": "UNKNOWN",
                "confidence": 0.5
            }

    def analyze(self, raw_text: str) -> Dict[str, Any]:
        """
        تحلیل یک متن ادبی و ارائه خروجی یکپارچه شامل:
          - نرمال‌سازی متن
          - تشخیص شعر یا نثر
          - تحلیل سبک ادبی
          - تحلیل وزن شعر (در صورت شعر بودن)
          - تشخیص آرایه‌های ادبی
          - شناسایی قالب شعری (در صورت شعر)

        Args:
            raw_text: متن ورودی ادبی

        Returns:
            دیکشنری یکپارچه شامل نتایج تحلیل.
        """
        normalized_text = self.normalizer.normalize(raw_text)
        normalized_text = re.sub(r'\n+', '\n', normalized_text).strip()
        is_poetry = self._is_poetry(normalized_text)
        style_result = self.analyze_style(normalized_text)
        meter_result = {}
        form_result = {}
        if is_poetry:
            meter_result = self.analyze_meter(normalized_text)
            form_result = self.identify_poetry_form(normalized_text, meter_result)
        devices_result = self.detect_devices(normalized_text)
        analysis_result = {
            "original_text": raw_text,
            "normalized_text": normalized_text,
            "is_poetry": is_poetry,
            "style_analysis": style_result,
            "meter_analysis": meter_result,
            "form_analysis": form_result,
            "device_analysis": devices_result,
            "timestamp": time.time()
        }
        return analysis_result


# تست نمونه
if __name__ == "__main__":
    sample_text = """هر آنکه آفتاب عشق در دل دارد،
از تاریکی شب بی‌خبر است.
در میان پرده‌های ظلمت،
نور امید تابان می‌شود."""
    analyzer = LiteratureAnalysis()
    result = analyzer.analyze(sample_text)
    import json

    print(json.dumps(result, ensure_ascii=False, indent=2))
