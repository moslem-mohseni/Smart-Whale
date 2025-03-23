"""
RequestHandler Module
-----------------------
این فایل مسئول دریافت و پردازش درخواست‌های ورودی در سیستم خودآموزی است.
این کلاس به عنوان یک روتینگ مرکزی عمل می‌کند که درخواست‌ها را بر اساس نوع (request_type)
به تابع‌های ثبت‌شده ارجاع می‌دهد. از مکانیزم‌های پیشرفته لاگینگ، کنترل خطا، و اجرا به صورت ناهمزمان استفاده می‌کند.
"""

import asyncio
import logging
from typing import Callable, Dict, Any, Optional, Coroutine, List

from .base_component import BaseComponent

# نوع تابع پردازش‌کننده درخواست: تابعی که یک دیکشنری دریافت کرده
# و یک coroutine بازمی‌گرداند که خروجی آن یک دیکشنری یا None است.
RequestHandlerFunction = Callable[[Dict[str, Any]], Coroutine[None, None, Optional[Dict[str, Any]]]]

class RequestHandler(BaseComponent):
    """
    کلاس RequestHandler به عنوان مرکز مدیریت درخواست‌های ورودی سیستم خودآموزی.

    ویژگی‌ها:
      - ثبت و لغو تابع‌های پردازش‌کننده درخواست بر اساس نوع درخواست.
      - پردازش درخواست‌ها به صورت ناهمزمان.
      - استفاده از لاگینگ، متریک‌ها و مکانیزم Circuit Breaker (در صورت نیاز) جهت افزایش پایداری.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(component_type="request_handler", config=config)
        # دیکشنری نگهدارنده لیست تابع‌های ثبت‌شده برای هر نوع درخواست
        self.handlers: Dict[str, List[RequestHandlerFunction]] = {}
        self.logger.info("[RequestHandler] Initialized.")

    def register_handler(self, request_type: str, handler: RequestHandlerFunction) -> None:
        """
        ثبت یک تابع پردازش‌کننده برای نوع مشخصی از درخواست.

        Args:
            request_type (str): نوع درخواست (مثلاً "FETCH_DATA", "TRAIN_MODEL" و ...)
            handler (RequestHandlerFunction): تابع پردازش‌کننده درخواست.
        """
        if request_type not in self.handlers:
            self.handlers[request_type] = []
        self.handlers[request_type].append(handler)
        self.logger.debug(f"[RequestHandler] Registered handler for request type: {request_type}")

    def unregister_handler(self, request_type: str, handler: RequestHandlerFunction) -> None:
        """
        لغو ثبت یک تابع پردازش‌کننده برای یک نوع درخواست.

        Args:
            request_type (str): نوع درخواست.
            handler (RequestHandlerFunction): تابع پردازش‌کننده برای حذف.
        """
        if request_type in self.handlers and handler in self.handlers[request_type]:
            self.handlers[request_type].remove(handler)
            self.logger.debug(f"[RequestHandler] Unregistered handler for request type: {request_type}")
            if not self.handlers[request_type]:
                del self.handlers[request_type]

    async def handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        پردازش یک درخواست ورودی و اجرای تابع‌های ثبت‌شده مطابق نوع درخواست.

        Args:
            request (Dict[str, Any]): دیکشنری حاوی اطلاعات درخواست، شامل کلید "type" به عنوان نوع درخواست.

        Returns:
            Optional[Dict[str, Any]]: خروجی پردازش درخواست (در صورت وجود)؛ در غیر این صورت None.
        """
        request_type = request.get("type")
        if not request_type:
            self.logger.error("[RequestHandler] Request missing 'type' field.")
            self.record_error_metric()
            return None

        self.logger.info(f"[RequestHandler] Handling request of type: {request_type}")
        handlers = self.handlers.get(request_type, [])
        if not handlers:
            self.logger.warning(f"[RequestHandler] No handlers registered for request type: {request_type}")
            return None

        results = []
        tasks = []
        # اجرای همه‌ی handlerهای ثبت‌شده به صورت ناهمزمان
        for handler in handlers:
            try:
                task = asyncio.create_task(handler(request))
                tasks.append(task)
            except Exception as e:
                self.logger.error(f"[RequestHandler] Error scheduling handler for {request_type}: {str(e)}")
                self.record_error_metric()

        if tasks:
            try:
                # اجرای تمام وظایف به صورت موازی
                results = await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                self.logger.error(f"[RequestHandler] Error executing handlers for {request_type}: {str(e)}")
                self.record_error_metric()
                return None

        # پردازش نتایج (می‌توان منطق ترکیب نتایج را اضافه کرد)
        valid_results = [res for res in results if not isinstance(res, Exception) and res is not None]
        self.logger.info(f"[RequestHandler] Completed handling request of type: {request_type} with {len(valid_results)} valid responses.")
        self.increment_metric(f"request_handled_{request_type}")
        return {"request_type": request_type, "responses": valid_results}


# مثال تستی برای RequestHandler
if __name__ == "__main__":
    async def sample_handler(request: Dict[str, Any]) -> Dict[str, Any]:
        # شبیه‌سازی پردازش درخواست
        await asyncio.sleep(0.5)
        return {"processed": True, "original": request}

    async def main():
        handler = RequestHandler()
        handler.register_handler("TEST_REQUEST", sample_handler)
        sample_request = {"type": "TEST_REQUEST", "data": {"key": "value"}}
        result = await handler.handle_request(sample_request)
        print("Request result:", result)

    asyncio.run(main())
