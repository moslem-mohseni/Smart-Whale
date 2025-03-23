import os


class BaseModelConfig:
    """
    مدیریت تنظیمات مدل پایه (ParsBERT) برای فاین‌تیونینگ.
    """

    # تنظیمات اصلی مدل
    MODEL_NAME = os.getenv("MODEL_NAME", "HooshvareLab/bert-base-parsbert-uncased")
    MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "models/base_model/saved/")
    USE_GPU = os.getenv("USE_GPU", "True").lower() == "true"

    # تنظیمات داده‌ها
    TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "data/processed/train.json")
    VALIDATION_DATA_PATH = os.getenv("VALIDATION_DATA_PATH", "data/processed/val.json")
    TEST_DATA_PATH = os.getenv("TEST_DATA_PATH", "data/processed/test.json")

    # پارامترهای آموزش
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", 3e-5))
    EPOCHS = int(os.getenv("EPOCHS", 5))
    GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", 1))
    WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 0.01))
    WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", 1000))

    # مسیرهای خروجی و لاگ‌ها
    LOGS_DIR = os.getenv("LOGS_DIR", "logs/base_model/")
    CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "models/base_model/checkpoints/")
    BEST_MODEL_PATH = os.getenv("BEST_MODEL_PATH", "models/base_model/best_model.pth")

    # تنظیمات سخت‌افزاری
    DEVICE = "cuda" if USE_GPU and os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

    @staticmethod
    def print_config():
        """ چاپ تمام تنظیمات مدل """
        config_vars = {key: value for key, value in BaseModelConfig.__dict__.items() if not key.startswith("__")}
        print("📌 تنظیمات مدل پایه ParsBERT:")
        for key, value in config_vars.items():
            print(f"🔹 {key}: {value}")


# ==================== تست ====================
if __name__ == "__main__":
    BaseModelConfig.print_config()
