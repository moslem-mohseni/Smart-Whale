import math
import random
from typing import List

class MathUtils:
    @staticmethod
    def mean(numbers: List[float]) -> float:
        """ محاسبه میانگین لیستی از اعداد """
        return sum(numbers) / len(numbers) if numbers else 0

    @staticmethod
    def median(numbers: List[float]) -> float:
        """ محاسبه میانه لیستی از اعداد """
        sorted_nums = sorted(numbers)
        n = len(sorted_nums)
        mid = n // 2
        return (sorted_nums[mid] if n % 2 != 0 else (sorted_nums[mid - 1] + sorted_nums[mid]) / 2) if numbers else 0

    @staticmethod
    def standard_deviation(numbers: List[float]) -> float:
        """ محاسبه انحراف معیار لیستی از اعداد """
        if not numbers:
            return 0
        mean_val = MathUtils.mean(numbers)
        variance = sum((x - mean_val) ** 2 for x in numbers) / len(numbers)
        return math.sqrt(variance)

    @staticmethod
    def random_number(min_val: int, max_val: int) -> int:
        """ تولید یک عدد تصادفی بین مقدار حداقل و حداکثر """
        return random.randint(min_val, max_val)

    @staticmethod
    def factorial(n: int) -> int:
        """ محاسبه فاکتوریل یک عدد """
        return math.factorial(n)

    @staticmethod
    def fibonacci(n: int) -> int:
        """ محاسبه عدد فیبوناچی در موقعیت n """
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

    @staticmethod
    def is_prime(n: int) -> bool:
        """ بررسی اینکه آیا یک عدد اول است یا خیر """
        if n < 2:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
