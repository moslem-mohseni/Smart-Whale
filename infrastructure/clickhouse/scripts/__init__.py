# infrastructure/clickhouse/scripts/__init__.py
"""
این ماژول شامل اسکریپت‌های مدیریتی و نگهداری ClickHouse است.
ابزارهایی برای بهینه‌سازی، پاکسازی و نگهداری سیستم در این بخش قرار دارند.
"""

from .maintenance import MaintenanceManager

__all__ = ['MaintenanceManager']