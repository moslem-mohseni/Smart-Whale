from typing import Dict, Any


class PriorityHandler:
    """
    مدیریت و تخصیص اولویت به درخواست‌ها برای بهینه‌سازی تخصیص منابع
    """

    def __init__(self):
        self.priority_levels = {
            "urgent": 3,
            "high": 2,
            "medium": 1,
            "low": 0
        }

    def evaluate_priority(self, request: Dict[str, Any]) -> int:
        """
        تعیین اولویت درخواست بر اساس نوع و نیاز پردازشی آن
        :param request: درخواست ورودی شامل اطلاعات مورد نیاز برای پردازش
        :return: سطح اولویت به صورت عددی (هرچه مقدار بیشتر باشد، اولویت بالاتر است)
        """
        request_type = request.get("type", "low").lower()
        return self.priority_levels.get(request_type, self.priority_levels["low"])

    def compare_priority(self, req1: Dict[str, Any], req2: Dict[str, Any]) -> int:
        """
        مقایسه دو درخواست بر اساس سطح اولویت آن‌ها
        :param req1: درخواست اول
        :param req2: درخواست دوم
        :return: 1 اگر اولویت req1 بیشتر باشد، -1 اگر req2 اولویت بیشتری داشته باشد، 0 در صورت برابری
        """
        priority1 = self.evaluate_priority(req1)
        priority2 = self.evaluate_priority(req2)

        if priority1 > priority2:
            return 1
        elif priority1 < priority2:
            return -1
        return 0


# نمونه استفاده از PriorityHandler برای تست
if __name__ == "__main__":
    priority_handler = PriorityHandler()
    request1 = {"type": "high"}
    request2 = {"type": "low"}

    print(f"Priority of request1: {priority_handler.evaluate_priority(request1)}")
    print(f"Priority of request2: {priority_handler.evaluate_priority(request2)}")
    print(f"Comparison result: {priority_handler.compare_priority(request1, request2)}")
