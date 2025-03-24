from typing import List
import numpy as np
from .models import Vector


class VectorOperations:
    """عملیات برداری برای Milvus"""

    @staticmethod
    def normalize(vector: Vector) -> Vector:
        """نرمال‌سازی بردار به واحد ۱"""
        norm = np.linalg.norm(vector.values)
        if norm == 0:
            return vector  # اگر نرمال صفر باشد، خود بردار بازگردانده شود.
        normalized_values = (np.array(vector.values) / norm).tolist()
        return Vector(id=vector.id, values=normalized_values, metadata=vector.metadata)

    @staticmethod
    def euclidean_distance(v1: Vector, v2: Vector) -> float:
        """محاسبه فاصله اقلیدسی بین دو بردار"""
        return np.linalg.norm(np.array(v1.values) - np.array(v2.values))

    @staticmethod
    def cosine_similarity(v1: Vector, v2: Vector) -> float:
        """محاسبه شباهت کسینوسی بین دو بردار"""
        vec1, vec2 = np.array(v1.values), np.array(v2.values)
        dot_product = np.dot(vec1, vec2)
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0  # اگر هر برداری نرمال صفر داشته باشد، شباهت ۰ است.
        return dot_product / (norm1 * norm2)

    @staticmethod
    def add_vectors(v1: Vector, v2: Vector) -> Vector:
        """جمع دو بردار"""
        new_values = (np.array(v1.values) + np.array(v2.values)).tolist()
        return Vector(id=f"{v1.id}_{v2.id}", values=new_values, metadata={**v1.metadata, **v2.metadata})

    @staticmethod
    def subtract_vectors(v1: Vector, v2: Vector) -> Vector:
        """تفریق دو بردار"""
        new_values = (np.array(v1.values) - np.array(v2.values)).tolist()
        return Vector(id=f"{v1.id}_sub_{v2.id}", values=new_values, metadata={**v1.metadata, **v2.metadata})
