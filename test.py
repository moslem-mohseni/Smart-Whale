

import torch
from teacher_model.parsbert_loader import load_parsbert
from student_model.model import ParsaModel


class TokenDecoder:
    def __init__(self):
        """ بارگذاری مدل دانش‌آموز و مدل معلم برای تولید متن فارسی """
        self.tokenizer, self.teacher_model = load_parsbert()
        self.student_model = ParsaModel(input_dim=768, output_dim=768)

    def decode(self, embedding):
        """ دریافت بردار و تبدیل آن به متن با کمک مدل دانش‌آموز و مدل معلم """

        # مدل پارسا سعی می‌کند خروجی را پیش‌بینی کند
        student_output = self.student_model(embedding)
        student_tokens = self.tokenizer.decode(torch.argmax(student_output, dim=-1))

        # بررسی میزان اطمینان مدل پارسا
        confidence = torch.softmax(student_output, dim=-1).max().item()

        if confidence >= 0.75:  # اگر مدل دانش‌آموز اطمینان کافی داشت
            return student_tokens
        else:
            # اگر مدل دانش‌آموز مطمئن نبود، از معلم کمک می‌گیرد
            teacher_tokens = self.tokenizer.decode(torch.argmax(self.teacher_model(embedding), dim=-1))

            # یادگیری از معلم: مدل پارسا خروجی معلم را ذخیره می‌کند
            self.student_model.learn_from_teacher(embedding, teacher_tokens)

            return teacher_tokens

