#!/bin/bash

# فعال کردن حالت خطایابی
set -euo pipefail
trap 'echo "Error on line $LINENO"' ERR

# تابع انتظار برای سرویس
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3

    echo "Waiting for $service at $host:$port..."
    while ! nc -z "$host" "$port"; do
        sleep 1
    done
    echo "$service is up"
}

# انتظار برای سرویس‌های اصلی
wait_for_service timescaledb 5432 "TimescaleDB"
wait_for_service redis 6379 "Redis"
wait_for_service kafka 9092 "Kafka"

# اجرای برنامه
echo "Starting application..."
exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4