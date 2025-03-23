import os
import json
import pandas as pd


class DataSaver:
    """
    کلاس مدیریت ذخیره داده‌های پردازش‌شده در فرمت JSON و CSV
    """

    def __init__(self, cache_dir="cache/social_media", cache_enabled=True):
        """
        :param cache_dir: مسیر ذخیره‌سازی کش
        :param cache_enabled: فعال/غیرفعال‌سازی کش
        """
        self.cache_dir = cache_dir
        self.cache_enabled = cache_enabled

        # ایجاد مسیر کش در صورت عدم وجود
        if cache_enabled and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def save_to_json(self, filename, data):
        """
        ذخیره داده‌ها در فایل JSON
        :param filename: نام فایل (بدون پسوند)
        :param data: داده‌های ذخیره‌شده
        """
        file_path = os.path.join(self.cache_dir, f"{filename}.json")
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    def load_from_json(self, filename):
        """
        بارگذاری داده‌ها از فایل JSON
        :param filename: نام فایل (بدون پسوند)
        :return: داده‌های ذخیره‌شده یا None در صورت عدم وجود فایل
        """
        file_path = os.path.join(self.cache_dir, f"{filename}.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        return None

    def save_to_csv(self, filename, data, columns=None):
        """
        ذخیره داده‌ها در فایل CSV
        :param filename: نام فایل (بدون پسوند)
        :param data: داده‌های ذخیره‌شده (لیست دیکشنری یا لیست رشته‌ها)
        :param columns: نام ستون‌ها (در صورت لیست دیکشنری)
        """
        file_path = os.path.join(self.cache_dir, f"{filename}.csv")
        df = pd.DataFrame(data, columns=columns) if isinstance(data, list) and isinstance(data[0],
                                                                                          dict) else pd.DataFrame(
            {"text": data})
        df.to_csv(file_path, index=False, encoding="utf-8-sig")

    def load_from_csv(self, filename):
        """
        بارگذاری داده‌ها از فایل CSV
        :param filename: نام فایل (بدون پسوند)
        :return: داده‌های ذخیره‌شده به صورت DataFrame یا None در صورت عدم وجود فایل
        """
        file_path = os.path.join(self.cache_dir, f"{filename}.csv")
        if os.path.exists(file_path):
            return pd.read_csv(file_path, encoding="utf-8-sig")
        return None


# ✅ تست داخلی برای بررسی عملکرد ذخیره و بازیابی داده‌ها
if __name__ == "__main__":
    saver = DataSaver(cache_enabled=True)

    test_data = ["این یک متن تستی است.", "پردازش داده‌های شبکه‌های اجتماعی.", "هوش مصنوعی در حال پیشرفت است."]

    print("✅ ذخیره و بارگذاری JSON:")
    saver.save_to_json("test_data", test_data)
    loaded_json = saver.load_from_json("test_data")
    print(loaded_json)

    print("\n✅ ذخیره و بارگذاری CSV:")
    saver.save_to_csv("test_data", test_data)
    loaded_csv = saver.load_from_csv("test_data")
    print(loaded_csv.head())
