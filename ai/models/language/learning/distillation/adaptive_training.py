import torch
import torch.nn.functional as F
from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB


class AdaptiveTraining:
    """
    مدیریت فرآیند یادگیری تطبیقی برای تنظیم پویا نرخ یادگیری و تأثیر Distillation.
    """

    def __init__(self, teacher_model, student_model, initial_temperature=2.0, initial_alpha=0.5):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = initial_temperature
        self.alpha = initial_alpha

        # اتصال به ClickHouse برای ذخیره اطلاعات فرآیند تطبیق یادگیری
        self.clickhouse_client = ClickHouseDB()

        # ایجاد جدول برای ذخیره متریک‌های یادگیری تطبیقی
        self.create_clickhouse_table()

    def create_clickhouse_table(self):
        """
        ایجاد جدول در ClickHouse برای ثبت اطلاعات فرآیند یادگیری تطبیقی.
        """
        query = """
        CREATE TABLE IF NOT EXISTS adaptive_training (
            teacher_model_name String,
            student_model_name String,
            version String,
            temperature Float32,
            alpha Float32,
            loss Float32,
            accuracy Float32,
            adjusted_lr Float32,
            timestamp DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY (teacher_model_name, timestamp);
        """
        self.clickhouse_client.execute_query(query)

    def adaptive_loss(self, teacher_logits, student_logits, labels, epoch, max_epochs):
        """
        تنظیم تطبیقی وزن Distillation و نرخ یادگیری بر اساس پیشرفت مدل.
        """
        progress = epoch / max_epochs
        dynamic_alpha = self.alpha * (1 - progress)
        dynamic_temperature = self.temperature * (1 - progress * 0.5)

        soft_targets = F.log_softmax(teacher_logits / dynamic_temperature, dim=1)
        soft_outputs = F.log_softmax(student_logits / dynamic_temperature, dim=1)

        kd_loss = F.kl_div(soft_outputs, soft_targets, reduction='batchmean') * (dynamic_temperature ** 2)
        ce_loss = F.cross_entropy(student_logits, labels)

        return (1 - dynamic_alpha) * ce_loss + dynamic_alpha * kd_loss, dynamic_alpha, dynamic_temperature

    def train_student(self, train_loader, optimizer, epochs=5, initial_lr=0.001):
        """
        اجرای فرآیند یادگیری تطبیقی مدل دانش‌آموز با انتقال دانش از مدل معلم.
        """
        best_accuracy = 0

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            # تنظیم نرخ یادگیری به‌صورت پویا
            adjusted_lr = initial_lr * (0.95 ** epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = adjusted_lr

            self.student_model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()

                with torch.no_grad():
                    teacher_logits = self.teacher_model(inputs)

                student_logits = self.student_model(inputs)
                loss, dynamic_alpha, dynamic_temperature = self.adaptive_loss(
                    teacher_logits, student_logits, labels, epoch, epochs
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                correct += (student_logits.argmax(1) == labels).sum().item()
                total += labels.size(0)

            epoch_loss = total_loss / len(train_loader)
            epoch_accuracy = correct / total

            print(
                f"🔹 Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Adjusted LR: {adjusted_lr:.6f}")

            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy

        # ثبت اطلاعات فرآیند یادگیری تطبیقی در ClickHouse
        self.log_adaptive_training_metrics(epoch_loss, best_accuracy, adjusted_lr)

    def log_adaptive_training_metrics(self, loss, accuracy, adjusted_lr):
        """
        ذخیره متریک‌های فرآیند یادگیری تطبیقی در ClickHouse.
        """
        query = f"""
        INSERT INTO adaptive_training (teacher_model_name, student_model_name, version, temperature, alpha, loss, accuracy, adjusted_lr)
        VALUES ('{self.teacher_model.__class__.__name__}', '{self.student_model.__class__.__name__}', '1.0', {self.temperature}, {self.alpha}, {loss}, {accuracy}, {adjusted_lr});
        """
        self.clickhouse_client.execute_query(query)

        print(f"✅ متریک‌های فرآیند یادگیری تطبیقی ثبت شد.")


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

    adaptive_training = AdaptiveTraining(teacher_model, student_model)

    # داده‌های ساختگی برای تست
    train_data = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)

    # اجرای فرآیند یادگیری تطبیقی
    adaptive_training.train_student(train_loader, optimizer, epochs=5)
