import torch
from ai.models.language.learning.trainer.data_preprocessor import DataPreprocessor
from ai.models.language.learning.trainer.fine_tuner import FineTuner
from ai.models.language.learning.trainer.model_updater import ModelUpdater
from ai.models.language.learning.trainer.distillation_manager import DistillationManager
from ai.models.language.learning.trainer.clickhouse_logger import ClickHouseLogger

class LearningPipeline:
    """
    مدیریت فرآیند یادگیری مدل‌ها، هماهنگ‌سازی تمام مراحل آموزش، بهینه‌سازی و Distillation.
    """

    def __init__(self, teacher_model, student_model):
        self.teacher_model = teacher_model
        self.student_model = student_model

        # اجزای مختلف یادگیری
        self.data_preprocessor = DataPreprocessor()
        self.fine_tuner = FineTuner(self.student_model)
        self.model_updater = ModelUpdater(self.student_model)
        self.distillation_manager = DistillationManager(teacher_model, student_model)
        self.logger = ClickHouseLogger()

    def preprocess_data(self, raw_data):
        """
        پردازش اولیه داده‌ها و حذف نویزها.
        """
        print("📌 در حال پردازش داده‌ها...")
        processed_data = [self.data_preprocessor.preprocess_text(text) for text in raw_data]
        return [text for text in processed_data if text is not None]  # حذف داده‌های تکراری

    def train_model(self, train_loader, val_loader, version="1.0"):
        """
        آموزش مدل دانش‌آموز و ذخیره نتایج فرآیند یادگیری.
        """
        print("📌 شروع آموزش مدل...")
        batch_size, learning_rate = self.fine_tuner.optimize_training(version)
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=learning_rate)

        self.fine_tuner.fine_tune(train_loader, val_loader, learning_rate, batch_size)

        print("📌 ثبت متریک‌های آموزشی در ClickHouse...")
        training_history = self.logger.get_training_history(self.student_model.__class__.__name__, version)
        return training_history

    def distill_knowledge(self, train_loader, optimizer, epochs=5):
        """
        اجرای فرآیند انتقال دانش از مدل معلم به مدل دانش‌آموز.
        """
        print("📌 شروع فرآیند Distillation...")
        self.distillation_manager.train_student(train_loader, optimizer, epochs)

    def update_model(self, version="1.1", accuracy=0.90, loss=0.05):
        """
        بررسی و بروزرسانی نسخه‌ی مدل دانش‌آموز در صورت بهبود عملکرد.
        """
        print("📌 بررسی و بروزرسانی مدل...")
        return self.model_updater.update_model(version, accuracy, loss)

    def execute_pipeline(self, raw_data, train_loader, val_loader, version="1.1"):
        """
        اجرای کامل فرآیند یادگیری: پردازش داده، آموزش مدل، انتقال دانش، و بروزرسانی نسخه مدل.
        """
        # مرحله ۱: پردازش داده‌ها
        processed_data = self.preprocess_data(raw_data)

        # مرحله ۲: آموزش مدل
        training_history = self.train_model(train_loader, val_loader, version)

        # مرحله ۳: Distillation (انتقال دانش از مدل معلم)
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=0.001)
        self.distill_knowledge(train_loader, optimizer)

        # مرحله ۴: بروزرسانی نسخه مدل
        latest_accuracy = training_history[0][2] if training_history else 0.90
        latest_loss = training_history[0][1] if training_history else 0.05
        self.update_model(version, latest_accuracy, latest_loss)

        print("✅ فرآیند یادگیری با موفقیت انجام شد.")

# تست عملکرد
if __name__ == "__main__":
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    class TeacherModel(nn.Module):
        def __init__(self):
            super(TeacherModel, self).__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    class StudentModel(nn.Module):
        def __init__(self):
            super(StudentModel, self).__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    teacher_model = TeacherModel()
    student_model = StudentModel()
    pipeline = LearningPipeline(teacher_model, student_model)

    # داده‌های خام تستی
    raw_data = ["این یک متن تستی است.", "مثال دیگر از داده‌های پردازشی."]

    # داده‌های ساختگی برای تست یادگیری
    train_data = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
    val_data = TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20,)))

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    # اجرای کامل فرآیند یادگیری
    pipeline.execute_pipeline(raw_data, train_loader, val_loader)
