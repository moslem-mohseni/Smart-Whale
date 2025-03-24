import os
import logging
import subprocess
from datetime import datetime

logger = logging.getLogger(__name__)

# متغیرهای محیطی برای مسیرهای پشتیبان‌گیری
BACKUP_DIR = os.getenv("TIMESCALEDB_BACKUP_DIR", "/var/backups/timescaledb")
DB_NAME = os.getenv("TIMESCALEDB_DATABASE", "timeseries_db")
DB_USER = os.getenv("TIMESCALEDB_USER", "db_user")
DB_HOST = os.getenv("TIMESCALEDB_HOST", "localhost")
DB_PORT = os.getenv("TIMESCALEDB_PORT", "5432")

def create_backup():
    """
    ایجاد یک نسخه پشتیبان از پایگاه داده
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"{BACKUP_DIR}/timescaledb_backup_{timestamp}.sql"

    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)

    try:
        logger.info(f"📦 در حال تهیه پشتیبان از پایگاه داده `{DB_NAME}`...")
        command = f"pg_dump -h {DB_HOST} -p {DB_PORT} -U {DB_USER} -F c -b -v -f {backup_file} {DB_NAME}"
        subprocess.run(command, shell=True, check=True, env={"PGPASSWORD": os.getenv("TIMESCALEDB_PASSWORD", "")})
        logger.info(f"✅ پشتیبان‌گیری موفقیت‌آمیز بود. فایل: {backup_file}")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ خطا در پشتیبان‌گیری: {e}")

def restore_backup(backup_file: str):
    """
    بازیابی پایگاه داده از یک نسخه پشتیبان

    Args:
        backup_file (str): مسیر فایل پشتیبان
    """
    try:
        logger.info(f"🔄 بازیابی پایگاه داده از `{backup_file}`...")
        command = f"pg_restore -h {DB_HOST} -p {DB_PORT} -U {DB_USER} -d {DB_NAME} -v {backup_file}"
        subprocess.run(command, shell=True, check=True, env={"PGPASSWORD": os.getenv("TIMESCALEDB_PASSWORD", "")})
        logger.info("✅ بازیابی پایگاه داده با موفقیت انجام شد.")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ خطا در بازیابی پایگاه داده: {e}")

# اجرای اسکریپت
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="پشتیبان‌گیری و بازیابی پایگاه داده TimescaleDB")
    parser.add_argument("--backup", action="store_true", help="ایجاد یک نسخه پشتیبان از پایگاه داده")
    parser.add_argument("--restore", type=str, help="بازیابی پایگاه داده از فایل مشخص شده")

    args = parser.parse_args()

    if args.backup:
        create_backup()
    elif args.restore:
        restore_backup(args.restore)
    else:
        print("❌ لطفاً یک گزینه مشخص کنید: `--backup` برای پشتیبان‌گیری یا `--restore <فایل>` برای بازیابی.")
