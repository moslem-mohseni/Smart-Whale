import torch
from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB

class IndependenceEvaluator:
    """
    سنجش میزان استقلال مدل دانش‌آموز از مدل معلم پس از یادگیری.
    """

    def __init__(self):
        self.clickhouse_client = ClickHouseDB()

        # ایجاد جدول برای ذخیره اطلاعات تحلیل استقلال مدل
        self.create_clickhouse_table()

    def create_clickhouse_table(self):
        """
        ایجاد جدول در ClickHouse برای ثبت داده‌های سنجش استقلال مدل دانش‌آموز.
        """
        query = """
        CREATE TABLE IF NOT EXISTS model_independence (
            student_model_name String,
            version String,
            accuracy Float32,
            divergence Float32,
            confidence_score Float32,
            timestamp DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY (student_model_name, timestamp);
        """
        self.clickhouse_client.execute_query(query)

    def compute_divergence(self, student_logits, teacher_logits):
        """
        محاسبه تفاوت خروجی مدل دانش‌آموز با مدل معلم (Kullback-Leibler Divergence).
        """
        student_probs = torch.softmax(student_logits, dim=1)
        teacher_probs = torch.softmax(teacher_logits, dim=1)
        divergence = torch.sum(teacher_probs * (torch.log(teacher_probs) - torch.log(student_probs)), dim=1).mean()
        return divergence.item()

    def evaluate_independence(self, student_model, teacher_model, data_loader):
        """
        ارزیابی مدل دانش‌آموز بدون وابستگی به مدل معلم و محاسبه میزان استقلال.
        """
        student_model.eval()
        teacher_model.eval()

        correct = 0
        total = 0
        total_divergence = 0
        total_confidence = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                student_logits = student_model(inputs)
                teacher_logits = teacher_model(inputs)

                predicted = torch.argmax(student_logits, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                total_divergence += self.compute_divergence(student_logits, teacher_logits)
                total_confidence += torch.max(torch.softmax(student_logits, dim=1), dim=1)[0].mean().item()

        accuracy = correct / total if total > 0 else 0
        avg_divergence = total_divergence / len(data_loader)
        avg_confidence = total_confidence / len(data_loader)

        return accuracy, avg_divergence, avg_confidence

    def log_independence_metrics(self, student_model_name, version, accuracy, divergence, confidence_score):
        """
        ذخیره متریک‌های تحلیل استقلال مدل در ClickHouse.
        """
        query = f"""
        INSERT INTO model_independence (student_model_name, version, accuracy, divergence, confidence_score)
        VALUES ('{student_model_name}', '{version}', {accuracy}, {divergence}, {confidence_score});
        """
        self.clickhouse_client.execute_query(query)

        print(f"✅ متریک‌های استقلال مدل {student_model_name} - نسخه {version} ثبت شد.")

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
    independence_evaluator = IndependenceEvaluator()

    # داده‌های ساختگی برای تست
    test_data = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # ارزیابی استقلال مدل
    accuracy, divergence, confidence_score = independence_evaluator.evaluate_independence(student_model, teacher_model, test_loader)

    # ثبت داده‌های استقلال مدل در ClickHouse
    independence_evaluator.log_independence_metrics("TestStudentModel", "1.0", accuracy, divergence, confidence_score)
