import random


class Config:
    """
    کلاس مدیریت تنظیمات Scraping برای توییتر و تلگرام
    """

    # ✅ تنظیمات پروکسی (در صورت نیاز برای جلوگیری از بلاک شدن)
    PROXIES = [
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8080",
    ]

    @staticmethod
    def get_random_proxy():
        """
        دریافت یک پروکسی تصادفی از لیست
        """
        return random.choice(Config.PROXIES)

    # ✅ تنظیمات هدرهای درخواست برای جلوگیری از شناسایی به عنوان بات
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }

    # ✅ محدودیت سرعت برای جلوگیری از بلاک شدن
    REQUEST_DELAY = 2  # تأخیر ۲ ثانیه بین درخواست‌ها


# ✅ تست داخلی برای بررسی تنظیمات
if __name__ == "__main__":
    print("✅ تست پروکسی:")
    print(Config.get_random_proxy())

    print("\n✅ تست User-Agent:")
    print(Config.HEADERS)
