# مرحله اول: محیط ساخت برای مدیریت پکیج‌ها
FROM python:3.11-slim as builder

# تنظیم متغیرهای محیطی برای بهینه‌سازی عملکرد pip و پایتون
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_TIMEOUT=1000 \
    PIP_DEFAULT_TIMEOUT=1000 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# نصب ابزارهای ضروری با حداقل حجم ممکن
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# کپی فایل‌های نیازمندی‌ها
COPY requirements.txt .
COPY requirements-dev.txt .

# به‌روزرسانی pip و نصب ابزارهای اساسی
RUN pip install --upgrade pip setuptools wheel

# ایجاد دایرکتوری wheels و دانلود پکیج‌ها با مدیریت خطا
RUN mkdir wheels && \
    # دانلود پکیج‌های اصلی به صورت جداگانه
    (pip download --no-deps --dest wheels \
        torch==2.5.1 \
        torchvision==0.20.1 \
        transformers==4.34.0 \
        filelock==3.12.4 \
        prometheus-client>=0.17.1 || true) && \
    # دانلود بقیه پکیج‌ها
    (pip download --no-deps --dest wheels -r requirements.txt || true) && \
    (pip download --no-deps --dest wheels -r requirements-dev.txt || true)

# ساخت wheels با مدیریت خطا و نادیده گرفتن وابستگی‌ها
RUN cd wheels && \
    for whl in *.whl *.tar.gz; do \
        if [ -f "$whl" ]; then \
            pip wheel --no-deps "$whl" || true; \
        fi \
    done

# مرحله دوم: ساخت ایمیج نهایی
FROM python:3.11-slim

# تنظیم متغیرهای محیطی
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:$PATH" \
    PIP_TIMEOUT=1000 \
    PIP_DEFAULT_TIMEOUT=1000

# نصب ابزارهای ضروری با حداقل حجم
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# ایجاد کاربر غیر root و دایرکتوری‌های مورد نیاز
RUN groupadd -r appgroup && \
    useradd -r -g appgroup -m -s /bin/bash appuser && \
    mkdir -p /app /scripts /app/logs /app/data /app/tmp /app/uploads && \
    chown -R appuser:appgroup /app /scripts

WORKDIR /app

# کپی wheels و فایل‌های نیازمندی‌ها
COPY --from=builder /build/wheels /wheels/
COPY --from=builder /build/requirements*.txt ./

# نصب پکیج‌ها با مدیریت خطا و ترتیب خاص
RUN pip install --upgrade pip && \
    # اولویت برای نصب prometheus-client
    (find /wheels -name "prometheus_client*.whl" -exec pip install --force-reinstall --no-deps {} \; || true) && \
    # نصب filelock به صورت جداگانه
    (find /wheels -name "filelock*.whl" -exec pip install --force-reinstall --no-deps {} \; || true) && \
    # نصب پکیج‌های اصلی
    (find /wheels -name "torch*.whl" -o -name "torchvision*.whl" -o -name "transformers*.whl" \
        -exec pip install --force-reinstall --no-deps {} \; || true) && \
    # نصب بقیه پکیج‌ها
    (for whl in /wheels/*.whl; do \
        if [ -f "$whl" ]; then \
            pip install --force-reinstall --no-deps "$whl" || true; \
        fi \
    done) && \
    # نصب مجدد prometheus-client برای اطمینان
    pip install prometheus-client>=0.17.1 && \
    # پاکسازی
    rm -rf /wheels/ /root/.cache/pip/*

# کپی کدهای برنامه و تنظیم دسترسی‌ها
COPY --chown=appuser:appgroup . .
COPY scripts/start.sh /scripts/

# تنظیم دسترسی‌های نهایی
RUN chmod +x /scripts/start.sh && \
    chown -R appuser:appgroup /scripts /app

USER appuser

# تنظیم health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["/scripts/start.sh"]