# استفاده از تصویر پایه پایتون
FROM python:3.9-slim

# تنظیم متغیرهای محیطی
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=1.1.11

# نصب ابزارهای مورد نیاز
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        netcat \
    && rm -rf /var/lib/apt/lists/*

# ایجاد و تنظیم دایرکتوری کاری
WORKDIR /app

# کپی فایل‌های مورد نیاز
COPY requirements.txt .
COPY . .

# نصب وابستگی‌ها
RUN pip install --no-cache-dir -r requirements.txt

# اسکریپت شروع برنامه
COPY ./scripts/start.sh /start.sh
RUN chmod +x /start.sh

# پورت پیش‌فرض
EXPOSE 8000

# دستور اجرای برنامه
CMD ["/start.sh"]