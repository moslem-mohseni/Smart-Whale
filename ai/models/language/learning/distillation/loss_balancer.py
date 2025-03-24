import torch
import torch.nn.functional as F
from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB

class LossBalancer:
    """
    مدیریت و تنظیم میزان تأثیر Loss مدل معلم و مدل دانش‌آموز برای دستیابی به تعادل بهینه.
    """

    def __init__(self, initial_alpha=0.5, min_alpha=0.1, decay_rate=0.05):
        self.alpha = initial_alpha
        self.min_alpha = min_alpha
        self.decay_rate = decay_rate

        # اتصال به ClickHouse برای ثبت داده‌های مربوط به میزان Loss
        self.clickhouse_client = ClickHouseDB()

        # ایجاد جدول برای ذخیره اطلاعات تنظیم Loss
        self.create_clickhouse_table()

    def create_clickhouse_table(self):
        """
        ایجاد جدول در ClickHouse برای ثبت اطلاعات تنظیم میزان تأثیر Loss مدل معلم و دانش‌آموز.
        """
        query = """
        CREATE TABLE IF NOT EXISTS loss_balancing (
            version String,
            epoch Int32,
            alpha Float32,
            loss Float32,
            adjusted_alpha Float32,
            timestamp DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY (version, timestamp);
        """
        self.clickhouse_client.execute_query(query)

    def balance_loss(self, teacher_logits, student_logits, labels, epoch, max_epochs):
        """
        تنظیم میزان تأثیر Loss بین مدل معلم و مدل دانش‌آموز به‌صورت پویا.
        """
        # کاهش تدریجی وابستگی به مدل معلم
        progress = epoch / max_epochs
        adjusted_alpha = max(self.min_alpha, self.alpha * (1 - self.decay_rate * epoch))

        soft_targets = F.log_softmax(teacher_logits, dim=1)
        soft_outputs = F.log_softmax(student_logits, dim=1)

        kd_loss = F.kl_div(soft_outputs, soft_targets, reduction='batchmean')
        ce_loss = F.cross_entropy(student_logits, labels)

        total_loss = (1 - adjusted_alpha) * ce_loss + adjusted_alpha * kd_loss

        return total_loss, adjusted_alpha

    def log_loss_balancing(self, version, epoch, loss, adjusted_alpha):
        """
        ذخیره متریک‌های فرآیند تنظیم Loss در ClickHouse.
        """
        query = f"""
        INSERT INTO loss_balancing (version, epoch, alpha, loss, adjusted_alpha)
        VALUES ('{version}', {epoch}, {self.alpha}, {loss}, {adjusted_alpha});
        """
        self.clickhouse_client.execute_query(query)

        print(f"✅ متریک‌های تنظیم Loss برای نسخه {version} - Epoch {epoch} ثبت شد.")

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
    loss_balancer = LossBalancer()

    # داده‌های ساختگی برای تست
    train_data = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)

    # اجرای فرآیند تنظیم Loss
    for epoch in range(5):
        for inputs, labels in train_loader:
            with torch.no_grad():
                teacher_logits = teacher_model(inputs)

            student_logits = student_model(inputs)
            loss, adjusted_alpha = loss_balancer.balance_loss(teacher_logits, student_logits, labels, epoch, 5)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_balancer.log_loss_balancing("1.0", epoch, loss.item(), adjusted_alpha)
