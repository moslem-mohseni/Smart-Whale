"""
مدل‌های داده‌ای برای دانش حوزه‌ای

این ماژول شامل مدل‌های داده‌ای مربوط به حوزه (Domain)، مفهوم (Concept)، رابطه (Relation) و ویژگی (Attribute) است.
استفاده از Data Classes باعث می‌شود کد تمیزتر، خواناتر و نگهداری آن آسان‌تر باشد.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any


def _format_datetime(dt: datetime) -> str:
    """تبدیل تاریخ به رشته با فرمت استاندارد"""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class Domain:
    domain_id: str
    domain_name: str
    domain_code: str
    parent_domain: Optional[str] = None
    description: Optional[str] = ""
    popularity: float = 0.0
    source: str = "default"
    discovery_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """تبدیل آبجکت به دیکشنری"""
        data = asdict(self)
        data["discovery_time"] = _format_datetime(self.discovery_time)
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Domain":
        """ایجاد آبجکت Domain از دیکشنری"""
        data_copy = data.copy()
        if "discovery_time" in data_copy and isinstance(data_copy["discovery_time"], str):
            data_copy["discovery_time"] = datetime.strptime(data_copy["discovery_time"], "%Y-%m-%d %H:%M:%S")
        return Domain(**data_copy)


@dataclass
class Concept:
    concept_id: str
    domain_id: str
    concept_name: str
    definition: str
    examples: List[str] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "default"
    usage_count: int = 1
    discovery_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """تبدیل آبجکت به دیکشنری"""
        data = asdict(self)
        data["discovery_time"] = _format_datetime(self.discovery_time)
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Concept":
        """ایجاد آبجکت Concept از دیکشنری"""
        data_copy = data.copy()
        if "discovery_time" in data_copy and isinstance(data_copy["discovery_time"], str):
            data_copy["discovery_time"] = datetime.strptime(data_copy["discovery_time"], "%Y-%m-%d %H:%M:%S")
        return Concept(**data_copy)


@dataclass
class Relation:
    relation_id: str
    source_concept_id: str
    target_concept_id: str
    relation_type: str
    description: str = ""
    confidence: float = 0.0
    source: str = "default"
    usage_count: int = 1
    discovery_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """تبدیل آبجکت به دیکشنری"""
        data = asdict(self)
        data["discovery_time"] = _format_datetime(self.discovery_time)
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Relation":
        """ایجاد آبجکت Relation از دیکشنری"""
        data_copy = data.copy()
        if "discovery_time" in data_copy and isinstance(data_copy["discovery_time"], str):
            data_copy["discovery_time"] = datetime.strptime(data_copy["discovery_time"], "%Y-%m-%d %H:%M:%S")
        return Relation(**data_copy)


@dataclass
class Attribute:
    attribute_id: str
    concept_id: str
    attribute_name: str
    attribute_value: str
    description: str = ""
    confidence: float = 0.0
    source: str = "default"
    usage_count: int = 1
    discovery_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """تبدیل آبجکت به دیکشنری"""
        data = asdict(self)
        data["discovery_time"] = _format_datetime(self.discovery_time)
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Attribute":
        """ایجاد آبجکت Attribute از دیکشنری"""
        data_copy = data.copy()
        if "discovery_time" in data_copy and isinstance(data_copy["discovery_time"], str):
            data_copy["discovery_time"] = datetime.strptime(data_copy["discovery_time"], "%Y-%m-%d %H:%M:%S")
        return Attribute(**data_copy)


# اگر نیاز باشد می‌توانیم یک تابع کمکی برای تبدیل لیست آبجکت‌ها به لیست دیکشنری ارائه کنیم:
def objects_to_dicts(objects: List[Any]) -> List[Dict[str, Any]]:
    return [obj.to_dict() for obj in objects]


# تست نمونه (در صورت اجرای مستقیم فایل)
if __name__ == "__main__":
    # نمونه ایجاد یک حوزه
    domain = Domain(
        domain_id="d_1001",
        domain_name="پزشکی",
        domain_code="MEDICAL",
        description="علوم پزشکی و درمانی",
        popularity=0.9,
        source="default"
    )
    print("Domain:", domain.to_dict())

    # نمونه ایجاد یک مفهوم
    concept = Concept(
        concept_id="c_1001_1",
        domain_id="d_1001",
        concept_name="آناتومی",
        definition="مطالعه ساختار بدن انسان",
        examples=["آناتومی سر", "آناتومی قلب"],
        confidence=0.95,
        source="default"
    )
    print("Concept:", concept.to_dict())

    # نمونه ایجاد یک رابطه
    relation = Relation(
        relation_id="r_1001",
        source_concept_id="c_1001_1",
        target_concept_id="c_1001_2",
        relation_type="IS_RELATED_TO",
        description="آناتومی و فیزیولوژی مرتبط هستند",
        confidence=0.9,
        source="default"
    )
    print("Relation:", relation.to_dict())

    # نمونه ایجاد یک ویژگی
    attribute = Attribute(
        attribute_id="a_1001",
        concept_id="c_1001_1",
        attribute_name="سطح دشواری",
        attribute_value="متوسط",
        description="سطح دشواری مطالعه آناتومی",
        confidence=0.8,
        source="default"
    )
    print("Attribute:", attribute.to_dict())
