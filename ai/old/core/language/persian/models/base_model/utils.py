import os
import random
import logging
import torch
import numpy as np
from models.base_model.config import BaseModelConfig


class Utils:
    """
    کلاس ابزارهای کمکی برای مدیریت پروژه.
    """

    @staticmethod
    def setup_logging(log_file="logs/base_model.log"):
        """
        تنظیمات لاگ‌گیری برای ذخیره خروجی‌ها در فایل و نمایش در کنسول.

        :param log_file: مسیر فایل لاگ
        """
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        logging.basicConfig(
            filename=log_file,
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO
        )

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger().addHandler(console)

        logging.info("✅ تنظیمات لاگ‌گیری انجام شد.")

    @staticmethod
    def ensure_dir(directory):
        """
        بررسی و ایجاد دایرکتوری در صورت عدم وجود.

        :param directory: مسیر دایرکتوری موردنظر
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def set_seed(seed=42):
        """
        تنظیم مقدار seed برای تولید مقادیر تصادفی، جهت تکرارپذیری آزمایش‌ها.

        :param seed: مقدار seed پیش‌فرض
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        logging.info(f"✅ مقدار Seed تنظیم شد: {seed}")

    @staticmethod
    def get_device():
        """
        دریافت دستگاه پردازشی (`cuda` یا `cpu`).

        :return: `torch.device`
        """
        device = torch.device("cuda" if BaseModelConfig.USE_GPU and torch.cuda.is_available() else "cpu")
        logging.info(f"✅ استفاده از دستگاه: {device}")
        return device

    @staticmethod
    def free_gpu_memory():
        """
        آزادسازی حافظه GPU در صورت لزوم.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("✅ حافظه GPU آزاد شد.")


# ==================== تست ====================
if __name__ == "__main__":
    Utils.setup_logging()
    Utils.set_seed(1234)
    Utils.ensure_dir("models/base_model/saved")
    device = Utils.get_device()
    Utils.free_gpu_memory()
