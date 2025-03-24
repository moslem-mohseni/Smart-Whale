import torch
import torch.nn.functional as F
from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB


class DistillationManager:
    """
    مدیریت فرآیند انتقال دانش (Knowledge Distillation) از مدل معلم به مدل دانش‌آموز.
    """

    def __init__(self, teacher_model, student_model, temperature=2.0, alpha=0.5):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha

        # استفاده از ClickHouse برای ثبت اطلاعات فرآیند Distillation
        self.clickhouse_client = ClickHouseDB()

        # ایجاد جدول برای ثبت عملکرد فرآیند Distillation
        self.create_clickhouse_table()

    def create_clickhouse_table(self):
        """
        ایجاد جدول در ClickHouse برای ذخیره متریک‌های Distillation.
        """
        query = """
        CREATE TABLE IF NOT EXISTS distillation_metrics (
            teacher_model_name String,
            student_model_name String,
            version String,
            temperature Float32,
            alpha Float32,
            loss Float32,
            accuracy Float32,
            timestamp DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY (teacher_model_name, timestamp);
        """
        self.clickhouse_client.execute_query(query)

    def knowledge_distillation_loss(self, teacher_logits, student_logits, labels):
        """
        محاسبه‌ی Loss برای فرآیند انتقال دانش.
        ترکیب Loss معمولی و Kullback-Leibler (KL) Divergence برای هماهنگ‌سازی خروجی‌های مدل معلم و دانش‌آموز.
        """
        soft_targets = F.log_softmax(teacher_logits / self.temperature, dim=1)
        soft_outputs = F.log_softmax(student_logits / self.temperature, dim=1)

        kd_loss = F.kl_div(soft_outputs, soft_targets, reduction='batchmean') * (self.temperature ** 2)
        ce_loss = F.cross_entropy(student_logits, labels)

        return (1 - self.alpha) * ce_loss + self.alpha * kd_loss

    def train_student(self, train_loader, optimizer, epochs=5):
        """
        اجرای فرآیند آموزش مدل دانش‌آموز با انتقال دانش از مدل معلم.
        """
        best_accuracy = 0

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            self.student_model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()

                with torch.no_grad():
                    teacher_logits = self.teacher_model(inputs)

                student_logits = self.student_model(inputs)
                loss = self.knowledge_distillation_loss(teacher_logits, student_logits, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                correct += (student_logits.argmax(1) == labels).sum().item()
                total += labels.size(0)

            epoch_loss = total_loss / len(train_loader)
            epoch_accuracy = correct / total

            print(f"🔹 Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy

        # ثبت اطلاعات فرآیند Distillation در ClickHouse
        self.log_distillation_metrics(epoch_loss, best_accuracy)

    def log_distillation_metrics(self, loss, accuracy):
        """
        ذخیره متریک‌های فرآیند Distillation در ClickHouse.
        """
        query = f"""
        INSERT INTO distillation_metrics (teacher_model_name, student_model_name, version, temperature, alpha, loss, accuracy)
        VALUES ('{self.teacher_model.__class__.__name__}', '{self.student_model.__class__.__name__}', '1.0', {self.temperature}, {self.alpha}, {loss}, {accuracy});
        """
        self.clickhouse_client.execute_query(query)

        print(f"✅ متریک‌های فرآیند Distillation ثبت شد.")


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

    distillation_manager = DistillationManager(teacher_model, student_model)

    # داده‌های ساختگی برای تست
    train_data = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)

    # اجرای فرآیند انتقال دانش
    distillation_manager.train_student(train_loader, optimizer)
