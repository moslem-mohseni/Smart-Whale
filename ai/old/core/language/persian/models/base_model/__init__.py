from .config import BaseModelConfig
from .dataset import PersianDataset, get_dataloader
from .model import PersianBERTModel
from .trainer import Trainer
from .evaluator import Evaluator
from .optimizer import OptimizerManager
from .loss import LossManager
from .feature_extractor import FeatureExtractor
from .embedding_manager import EmbeddingManager
from .utils import Utils

__all__ = [
    "BaseModelConfig",
    "PersianDataset",
    "get_dataloader",
    "PersianBERTModel",
    "Trainer",
    "Evaluator",
    "OptimizerManager",
    "LossManager",
    "FeatureExtractor",
    "EmbeddingManager",
    "Utils"
]


# =========================== TEST ===========================
if __name__ == "__main__":
    print("📌 مقداردهی اولیه `models/base_model/` با موفقیت انجام شد.")

    # تست مقداردهی اولیه کلاس‌ها
    config = BaseModelConfig()
    trainer = Trainer(num_labels=3)
    evaluator = Evaluator(num_labels=3)
    optimizer_manager = OptimizerManager(trainer.model)
    loss_manager = LossManager()
    feature_extractor = FeatureExtractor()
    embedding_manager = EmbeddingManager()

    print("✅ تمامی ماژول‌ها مقداردهی شدند و آماده استفاده هستند.")
