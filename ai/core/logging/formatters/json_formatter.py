import json
import logging
from datetime import datetime

class JSONLogFormatter(logging.Formatter):
    def format(self, record):
        """
        فرمت کردن لاگ‌ها به‌صورت JSON برای پردازش بهتر
        :param record: رکورد لاگ
        :return: رشته JSON فرمت‌شده
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage()
        }

        # افزودن اطلاعات اضافی در صورت موجود بودن
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_entry.update(record.extra)

        return json.dumps(log_entry, ensure_ascii=False)
