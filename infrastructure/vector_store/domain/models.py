from typing import List, Dict, Any
import json


class Vector:
    """مدل داده‌ای برای بردارها در Milvus"""

    def __init__(self, id: str, values: List[float], metadata: Dict[str, Any] = None):
        """
        مقداردهی اولیه بردار
        :param id: شناسه منحصربه‌فرد بردار
        :param values: لیست مقادیر عددی بردار
        :param metadata: اطلاعات اضافی مربوط به بردار
        """
        self.id = id
        self.values = values
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """تبدیل بردار به فرمت دیکشنری"""
        return {
            "id": self.id,
            "values": self.values,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """ایجاد نمونه‌ای از Vector از دیکشنری"""
        return cls(
            id=data["id"],
            values=data["values"],
            metadata=data.get("metadata", {})
        )

    def to_json(self) -> str:
        """تبدیل بردار به JSON"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str):
        """ایجاد نمونه‌ای از Vector از JSON"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __repr__(self):
        return f"Vector(id={self.id}, dim={len(self.values)}, metadata={self.metadata})"
