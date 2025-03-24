import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import logging
from config import CONFIG


class SmartModel(nn.Module):
    """
    مدل یادگیری زبان پیشرفته که به مرور از مدل معلم یاد گرفته و به استقلال می‌رسد.
    این مدل تکاملی با گذشت زمان، وابستگی به مدل معلم را کاهش داده و به صورت مستقل عمل می‌کند.
    """

    def __init__(self, input_dim=768, hidden_dim=2048, output_dim=768):
        super(SmartModel, self).__init__()

        # معماری عمیق و پیشرفته برای یادگیری بهتر
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim * 3),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim * 3),
            nn.Dropout(0.3)
        )

        self.processor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 3),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

        # مدل‌های تخصصی برای وظایف مختلف
        self.intent_classifier = nn.Sequential(
            nn.Linear(output_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 5)  # 5 کلاس نیت
        )

        self.semantic_analyzer = nn.Sequential(
            nn.Linear(output_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        self.grammar_analyzer = nn.Sequential(
            nn.Linear(output_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # بهینه‌سازها
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999))
        self.criterion = nn.MSELoss()

        # ردیابی پیشرفت
        self.confidence_scores = {}  # ذخیره امتیاز اطمینان برای هر نوع داده
        self.training_metrics = {
            "dialect_learning": [],
            "grammar_learning": [],
            "semantic_learning": [],
            "literary_learning": [],
            "proverb_learning": []
        }

        # شمارنده‌های یادگیری
        self.learning_counts = {
            "total": 0,
            "dialect": 0,
            "grammar": 0,
            "semantic": 0,
            "literary": 0,
            "proverb": 0,
            "domain": 0,
            "vector": 0,
            "graph": 0
        }

    def forward(self, x):
        """
        پردازش ورودی از طریق شبکه عصبی
        """
        # تبدیل ورودی به تنسور اگر نیست
        if not isinstance(x, torch.Tensor):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            elif isinstance(x, list):
                x = torch.tensor(x, dtype=torch.float32)
            else:
                # شبیه‌سازی تبدیل متن به بردار
                x = torch.randn(768)  # بردار تصادفی به جای پردازش واقعی

        # اطمینان از شکل صحیح
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # افزودن بعد بسته

        # عبور از شبکه
        encoded = self.encoder(x)
        processed = self.processor(encoded)
        output = self.decoder(processed)

        return output

    def confidence_level(self, text):
        """
        تعیین سطح اطمینان مدل به توانایی خود در پردازش متن
        """
        # شبیه‌سازی تبدیل متن به بردار
        if isinstance(text, str):
            text_key = hash(text) % 10000  # کلید منحصربفرد برای هر متن

            # اگر قبلاً این متن را دیده‌ایم، اطمینان را افزایش می‌دهیم
            if text_key in self.confidence_scores:
                self.confidence_scores[text_key] = min(
                    self.confidence_scores[text_key] + 0.05,
                    0.99
                )
                return self.confidence_scores[text_key]

            # برای متن‌های جدید، اطمینان را بر اساس یادگیری کلی محاسبه می‌کنیم
            base_confidence = 0.4  # اطمینان پایه
            learning_factor = min(self.learning_counts["total"] / 1000, 0.5)  # حداکثر 0.5 افزایش با یادگیری

            # محاسبه اطمینان بر اساس تعداد یادگیری‌ها
            confidence = base_confidence + learning_factor

            # ذخیره اطمینان برای استفاده بعدی
            self.confidence_scores[text_key] = confidence
            return confidence

        elif isinstance(text, torch.Tensor) or isinstance(text, np.ndarray):
            # برای ورودی‌های برداری، یک مقدار اطمینان شبیه‌سازی‌شده برمی‌گردانیم
            with torch.no_grad():
                # محاسبه اطمینان بر اساس نرم بردار
                if isinstance(text, np.ndarray):
                    norm_val = np.linalg.norm(text)
                else:
                    norm_val = torch.norm(text).item()

                # اطمینان بر اساس نرم و میزان یادگیری
                base_confidence = 0.4
                norm_factor = min(norm_val / 100, 0.3)  # حداکثر 0.3 از نرم
                learning_factor = min(self.learning_counts["total"] / 1000, 0.3)  # حداکثر 0.3 از یادگیری

                return base_confidence + norm_factor + learning_factor

        return 0.5  # مقدار پیش‌فرض برای موارد نامشخص

    def train_model(self, input_data, target_data):
        """
        آموزش مدل بر اساس داده‌های معلم تا رسیدن به استقلال.
        """
        # تبدیل داده‌ها به تنسور اگر نیستند
        if not isinstance(input_data, torch.Tensor):
            if isinstance(input_data, np.ndarray):
                input_data = torch.from_numpy(input_data).float()
            elif isinstance(input_data, list):
                input_data = torch.tensor(input_data, dtype=torch.float32)
            else:
                # شبیه‌سازی تبدیل متن به بردار
                input_data = torch.randn(768)

        if not isinstance(target_data, torch.Tensor):
            if isinstance(target_data, np.ndarray):
                target_data = torch.from_numpy(target_data).float()
            elif isinstance(target_data, list):
                target_data = torch.tensor(target_data, dtype=torch.float32)
            else:
                # شبیه‌سازی تبدیل هدف به بردار
                target_data = torch.randn(768)

        # اطمینان از شکل صحیح
        if len(input_data.shape) == 1:
            input_data = input_data.unsqueeze(0)
        if len(target_data.shape) == 1:
            target_data = target_data.unsqueeze(0)

        # آموزش مدل
        self.train()  # حالت آموزش
        self.optimizer.zero_grad()
        output = self.forward(input_data)
        loss = self.criterion(output, target_data)
        loss.backward()
        self.optimizer.step()

        # افزایش شمارنده یادگیری کلی
        self.learning_counts["total"] += 1

        return loss.item()

    def learn_from_teacher(self, text, teacher_output):
        """
        یادگیری از خروجی مدل معلم
        """
        # شبیه‌سازی تبدیل متن به بردار
        text_vector = torch.randn(768)  # در واقعیت باید از یک توکنایزر و مدل مناسب استفاده شود

        # تبدیل خروجی معلم به تنسور اگر نیست
        if not isinstance(teacher_output, torch.Tensor):
            if isinstance(teacher_output, np.ndarray):
                teacher_output = torch.from_numpy(teacher_output).float()
            elif isinstance(teacher_output, list):
                teacher_output = torch.tensor(teacher_output, dtype=torch.float32)
            elif isinstance(teacher_output, dict):
                # پردازش خروجی‌های مختلف معلم
                if "embedding" in teacher_output:
                    teacher_output = torch.tensor(teacher_output["embedding"], dtype=torch.float32)
                else:
                    # اگر بردار موجود نباشد، فقط متا‌دیتا را ذخیره می‌کنیم
                    logging.info(f"یادگیری داده‌های غیر برداری: {teacher_output}")
                    self.learning_counts["semantic"] += 1
                    return 0.0
            elif isinstance(teacher_output, str):
                # یادگیری از خروجی‌های متنی (مثل نیت)
                self.learning_counts["semantic"] += 1
                return 0.0

        # افزایش شمارنده یادگیری معنایی
        self.learning_counts["semantic"] += 1

        # آموزش مدل
        return self.train_model(text_vector, teacher_output)

    def learn_dialect(self, standard_word, dialect_word, dialect, example_sentence=None):
        """
        یادگیری لهجه‌های مختلف فارسی
        """
        # شبیه‌سازی یادگیری لهجه
        self.learning_counts["dialect"] += 1

        # ذخیره متریک‌های یادگیری
        self.training_metrics["dialect_learning"].append({
            "standard": standard_word,
            "dialect": dialect_word,
            "dialect_type": dialect,
            "timestamp": torch.cuda.current_device() if torch.cuda.is_available() else 0
            # در اینجا فقط به عنوان شمارنده زمان
        })

        # شبیه‌سازی آموزش مدل
        with torch.no_grad():
            input_vector = torch.randn(768)
            target_vector = torch.randn(768)

        return self.train_model(input_vector, target_vector)

    def analyze_grammar(self, text):
        """
        تحلیل گرامری متن فارسی
        """
        # شبیه‌سازی تحلیل گرامری
        self.eval()  # حالت ارزیابی

        with torch.no_grad():
            # تبدیل متن به بردار (شبیه‌سازی)
            text_vector = torch.randn(768)

            # عبور از مدل
            embedding = self.forward(text_vector)
            grammar_features = self.grammar_analyzer(embedding)

            # شبیه‌سازی خروجی تحلیل گرامری
            words = text.split()
            result = []

            # هر چند کلمه، یک خطای گرامری شبیه‌سازی می‌کنیم
            for i, word in enumerate(words):
                if i % 5 == 0 and len(word) > 3:  # فقط برای شبیه‌سازی
                    result.append({
                        "word": word,
                        "suggested": word.replace(word[-1], 'ه') if word[-1] != 'ه' else word + 'ی',
                        "error_type": "Misconjugation"
                    })

            return result

    def learn_literary_work(self, title, author, style=None):
        """
        یادگیری آثار ادبی فارسی
        """
        # شبیه‌سازی یادگیری اثر ادبی
        self.learning_counts["literary"] += 1

        # ذخیره متریک‌های یادگیری
        self.training_metrics["literary_learning"].append({
            "title": title,
            "author": author,
            "style": style,
            "timestamp": torch.cuda.current_device() if torch.cuda.is_available() else 0
        })

        # شبیه‌سازی آموزش مدل
        with torch.no_grad():
            input_vector = torch.randn(768)
            target_vector = torch.randn(768)

        return self.train_model(input_vector, target_vector)

    def learn_proverb(self, proverb, meaning):
        """
        یادگیری ضرب‌المثل‌های فارسی
        """
        # شبیه‌سازی یادگیری ضرب‌المثل
        self.learning_counts["proverb"] += 1

        # ذخیره متریک‌های یادگیری
        self.training_metrics["proverb_learning"].append({
            "proverb": proverb,
            "meaning": meaning,
            "timestamp": torch.cuda.current_device() if torch.cuda.is_available() else 0
        })

        # شبیه‌سازی آموزش مدل
        with torch.no_grad():
            input_vector = torch.randn(768)
            target_vector = torch.randn(768)

        return self.train_model(input_vector, target_vector)

    def learn_domain_concept(self, domain, concept, parent=None):
        """
        یادگیری مفاهیم تخصصی در حوزه‌های مختلف
        """
        # شبیه‌سازی یادگیری مفهوم تخصصی
        self.learning_counts["domain"] += 1

        # شبیه‌سازی آموزش مدل
        with torch.no_grad():
            input_vector = torch.randn(768)
            target_vector = torch.randn(768)

        return self.train_model(input_vector, target_vector)

    def learn_vector_embedding(self, concept, vector):
        """
        یادگیری بردارهای معنایی مفاهیم
        """
        # شبیه‌سازی یادگیری بردار معنایی
        self.learning_counts["vector"] += 1

        # تبدیل بردار به تنسور
        if not isinstance(vector, torch.Tensor):
            if isinstance(vector, np.ndarray):
                vector = torch.from_numpy(vector).float()
            elif isinstance(vector, list):
                vector = torch.tensor(vector, dtype=torch.float32)
            else:
                vector = torch.randn(768)

        # شبیه‌سازی آموزش مدل
        with torch.no_grad():
            input_vector = torch.randn(768)

        return self.train_model(input_vector, vector)

    def learn_graph_relation(self, concept1, concept2, relation_type):
        """
        یادگیری روابط گراف دانش
        """
        # شبیه‌سازی یادگیری رابطه گراف
        self.learning_counts["graph"] += 1

        # شبیه‌سازی آموزش مدل
        with torch.no_grad():
            input_vector = torch.randn(768)
            target_vector = torch.randn(768)

        return self.train_model(input_vector, target_vector)

    def detect_intent(self, text):
        """
        تشخیص هدف جمله
        """
        self.eval()  # حالت ارزیابی

        with torch.no_grad():
            # تبدیل متن به بردار (شبیه‌سازی)
            text_vector = torch.randn(768)

            # عبور از مدل
            embedding = self.forward(text_vector)
            intent_scores = self.intent_classifier(embedding)
            intent_probs = torch.softmax(intent_scores, dim=-1)

            # شناسایی بالاترین احتمال
            intent_index = torch.argmax(intent_probs).item()
            intents = ["greeting", "farewell", "question", "statement", "unknown"]

            return intents[intent_index] if intent_index < len(intents) else "unknown"

    def analyze_semantics(self, text):
        """
        تحلیل معنایی جامع متن
        """
        self.eval()  # حالت ارزیابی

        with torch.no_grad():
            # تبدیل متن به بردار (شبیه‌سازی)
            text_vector = torch.randn(768)

            # عبور از مدل
            embedding = self.forward(text_vector)
            semantic_features = self.semantic_analyzer(embedding)

            # شبیه‌سازی تحلیل معنایی
            intent = self.detect_intent(text)

            # تحلیل احساسات (شبیه‌سازی)
            sentiment_score = torch.sum(semantic_features).item()
            sentiment = "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"

            # استخراج موضوعات (شبیه‌سازی)
            topics = ["فناوری", "سلامت", "آموزش"] if len(text) > 30 else ["عمومی"]

            return {
                "intent": intent,
                "sentiment": sentiment,
                "topics": topics,
                "embedding": semantic_features.cpu().numpy().tolist()
            }

    def analyze_literary_style(self, text):
        """
        تحلیل سبک ادبی متن
        """
        self.eval()  # حالت ارزیابی

        with torch.no_grad():
            # شبیه‌سازی تحلیل سبک ادبی
            styles = ["FORMAL", "INFORMAL", "COLLOQUIAL", "TECHNICAL_PROSE", "JOURNALISTIC"]

            # سبک بر اساس طول متن (فقط برای شبیه‌سازی)
            text_length = len(text)

            if text_length > 200:
                return styles[0]  # FORMAL
            elif text_length > 150:
                return styles[3]  # TECHNICAL_PROSE
            elif text_length > 100:
                return styles[4]  # JOURNALISTIC
            elif text_length > 50:
                return styles[2]  # COLLOQUIAL
            else:
                return styles[1]  # INFORMAL

    def get_learning_statistics(self):
        """
        دریافت آمار یادگیری مدل
        """
        return {
            "learning_counts": self.learning_counts,
            "average_confidence": sum(self.confidence_scores.values()) / len(
                self.confidence_scores) if self.confidence_scores else 0,
            "confidence_samples": len(self.confidence_scores)
        }

    def save_model(self, path=CONFIG["smart_model"]):
        """
        ذخیره مدل یادگیرنده Smart Whale به همراه آمار یادگیری
        """
        state_dict = self.state_dict()

        # اضافه کردن آمار یادگیری
        learning_data = {
            "confidence_scores": {str(k): v for k, v in self.confidence_scores.items()},
            "learning_counts": self.learning_counts,
            "training_metrics": {k: v[:100] for k, v in self.training_metrics.items()}  # محدود کردن تعداد لاگ‌ها
        }

        save_data = {
            "state_dict": state_dict,
            "learning_data": learning_data
        }

        torch.save(save_data, path)
        logging.info(f"مدل Smart Whale با موفقیت در مسیر {path} ذخیره شد.")

    def load_model(self, path=CONFIG["smart_model"]):
        """
        بارگذاری مدل از فایل ذخیره‌شده به همراه آمار یادگیری
        """
        try:
            save_data = torch.load(path)

            # بارگذاری وزن‌های مدل
            if "state_dict" in save_data:
                self.load_state_dict(save_data["state_dict"])
            else:
                # سازگاری با فرمت قدیمی
                self.load_state_dict(save_data)

            # بارگذاری آمار یادگیری
            if "learning_data" in save_data:
                learning_data = save_data["learning_data"]

                if "confidence_scores" in learning_data:
                    self.confidence_scores = {int(k): v for k, v in learning_data["confidence_scores"].items()}

                if "learning_counts" in learning_data:
                    self.learning_counts = learning_data["learning_counts"]

                if "training_metrics" in learning_data:
                    self.training_metrics = learning_data["training_metrics"]

            logging.info(f"مدل Smart Whale با موفقیت از مسیر {path} بارگذاری شد.")

        except Exception as e:
            logging.error(f"خطا در بارگذاری مدل: {e}")
