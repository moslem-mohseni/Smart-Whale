import asyncio
from typing import Dict, Any
from .load_balancer import LoadBalancer
from .priority_handler import PriorityHandler

class RequestDispatcher:
    """
    مدیریت و توزیع درخواست‌های ورودی به مدل‌های مناسب
    """
    def __init__(self):
        self.load_balancer = LoadBalancer()
        self.priority_handler = PriorityHandler()

    async def dispatch_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        دریافت درخواست و مسیریابی آن به مدل مناسب
        :param request: درخواست ورودی شامل اطلاعات مربوط به پردازش
        :return: پاسخ مدل پردازشی
        """
        try:
            # 1️⃣ بررسی اولویت درخواست
            priority = self.priority_handler.evaluate_priority(request)
            request["priority"] = priority

            # 2️⃣ تخصیص مسیر مناسب از طریق متعادل‌سازی بار
            selected_model = await self.load_balancer.select_model(request)

            if not selected_model:
                return {"status": "error", "message": "No available model to process the request."}

            # 3️⃣ ارسال درخواست به مدل انتخاب‌شده
            response = await self._send_to_model(selected_model, request)
            return response

        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _send_to_model(self, model: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        ارسال درخواست به مدل انتخاب‌شده و دریافت پاسخ
        :param model: نام مدل انتخاب‌شده
        :param request: درخواست پردازشی
        :return: نتیجه پردازش مدل
        """
        await asyncio.sleep(0.1)  # شبیه‌سازی تأخیر در ارتباط
        return {"status": "success", "model": model, "result": f"Processed by {model}"}

# نمونه استفاده از RequestDispatcher برای تست
if __name__ == "__main__":
    dispatcher = RequestDispatcher()
    test_request = {"data": "sample_input", "type": "classification"}
    response = asyncio.run(dispatcher.dispatch_request(test_request))
    print(response)
