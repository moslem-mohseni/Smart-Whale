import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from hazm import Normalizer
from config import CONFIG
import numpy as np
import os
import logging


class TeacherModel:
    """
    مدل معلم پیشرفته که با استفاده از ParsBERT، دانش اولیه پردازش زبان فارسی را فراهم می‌کند.
    این مدل به عنوان منبع اصلی دانش برای SmartModel عمل کرده و به تدریج نقش خود را به آن واگذار می‌کند.
    """

    def __init__(self):
        """راه‌اندازی مدل معلم با بارگذاری ParsBERT و ابزارهای مورد نیاز"""
        self.normalizer = Normalizer()
        self.parsbert = self._load_parsbert()
        self.tokenizer = self._load_tokenizer()

        # مدل‌های تخصصی برای پردازش‌های مختلف
        self.intent_classifier = self._load_intent_classifier()
        self.grammar_analyzer = self._load_grammar_analyzer()
        self.semantic_analyzer = self._load_semantic_analyzer()

        # حالت آماده‌باش
        self.ready = self.parsbert is not None

    def _load_parsbert(self):
        """بارگذاری مدل ParsBERT"""
        try:
            model = AutoModel.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
            return model
        except Exception as e:
            logging.error(f"خطا در بارگذاری ParsBERT: {e}")
            return None

    def _load_tokenizer(self):
        """بارگذاری توکنایزر مدل"""
        try:
            tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
            return tokenizer
        except Exception as e:
            logging.error(f"خطا در بارگذاری توکنایزر: {e}")
            return None

    def _load_intent_classifier(self):
        """بارگذاری مدل دسته‌بندی نیت"""
        try:
            # شبیه‌سازی مدل دسته‌بندی نیت با یک شبکه عصبی ساده
            model = nn.Sequential(
                nn.Linear(768, 1024),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Linear(512, 5)  # 5 کلاس نیت
            )
            # در صورت وجود فایل مدل، آن را بارگذاری می‌کنیم
            if os.path.exists("intent_classifier.pth"):
                model.load_state_dict(torch.load("intent_classifier.pth"))
            return model
        except Exception as e:
            logging.error(f"خطا در بارگذاری مدل دسته‌بندی نیت: {e}")
            return None

    def _load_grammar_analyzer(self):
        """بارگذاری مدل تحلیل گرامری"""
        try:
            # شبیه‌سازی مدل تحلیل گرامری
            model = nn.Sequential(
                nn.Linear(768, 1536),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(1536, 2048),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(2048, 1024),
                nn.Tanh(),
                nn.Linear(1024, 768)
            )
            if os.path.exists("grammar_analyzer.pth"):
                model.load_state_dict(torch.load("grammar_analyzer.pth"))
            return model
        except Exception as e:
            logging.error(f"خطا در بارگذاری مدل تحلیل گرامری: {e}")
            return None

    def _load_semantic_analyzer(self):
        """بارگذاری مدل تحلیل معنایی"""
        try:
            # شبیه‌سازی مدل تحلیل معنایی
            model = nn.Sequential(
                nn.Linear(768, 2048),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(2048, 3072),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(3072, 2048),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(2048, 1024),
                nn.Tanh(),
                nn.Linear(1024, 768)
            )
            if os.path.exists("semantic_analyzer.pth"):
                model.load_state_dict(torch.load("semantic_analyzer.pth"))
            return model
        except Exception as e:
            logging.error(f"خطا در بارگذاری مدل تحلیل معنایی: {e}")
            return None

    def _preprocess_text(self, text):
        """پیش‌پردازش متن ورودی"""
        if not self.ready:
            return None

        normalized_text = self.normalizer.normalize(text)
        inputs = self.tokenizer(normalized_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        return inputs

    def forward(self, text):
        """
        پردازش اصلی متن توسط مدل معلم
        """
        if not self.ready:
            return torch.zeros(768)

        inputs = self._preprocess_text(text)
        with torch.no_grad():
            outputs = self.parsbert(**inputs)
            # میانگین‌گیری از تمام توکن‌ها برای داشتن یک بردار نمایشی از کل متن
            embedding = outputs.last_hidden_state.mean(dim=1)

        return embedding.detach()

    def detect_intent(self, text):
        """تشخیص هدف جمله با استفاده از ParsBERT"""
        if not self.ready or self.intent_classifier is None:
            return "unknown"

        inputs = self._preprocess_text(text)
        with torch.no_grad():
            outputs = self.parsbert(**inputs)
            features = outputs.pooler_output
            intent_scores = self.intent_classifier(features)
            intent_probs = torch.softmax(intent_scores, dim=-1)

            # شناسایی بالاترین احتمال
            intent_index = torch.argmax(intent_probs).item()
            intents = ["greeting", "farewell", "question", "statement", "unknown"]

            return intents[intent_index] if intent_index < len(intents) else "unknown"

    def analyze_grammar(self, text):
        """تحلیل گرامری متن با استفاده از مدل معلم"""
        if not self.ready or self.grammar_analyzer is None:
            return []

        inputs = self._preprocess_text(text)
        text_tokens = self.tokenizer.tokenize(self.normalizer.normalize(text))

        with torch.no_grad():
            outputs = self.parsbert(**inputs)
            token_embeddings = outputs.last_hidden_state

            # شبیه‌سازی تحلیل گرامری
            features = self.grammar_analyzer(token_embeddings.mean(dim=1))

            # تشخیص خطاهای گرامری (شبیه‌سازی)
            errors = []
            for i, token in enumerate(text_tokens[:min(len(text_tokens), 50)]):
                if i % 7 == 0:  # فقط برای شبیه‌سازی، هر هفتمین توکن را به عنوان خطا در نظر می‌گیریم
                    errors.append({
                        "word": token,
                        "suggested": token + "تر" if len(token) > 2 else token,
                        "error_type": "Misconjugation"
                    })

            return errors

    def analyze_semantics(self, text):
        """تحلیل معنایی جامع متن"""
        if not self.ready or self.semantic_analyzer is None:
            return {"intent": "unknown", "sentiment": "neutral", "topics": []}

        inputs = self._preprocess_text(text)

        with torch.no_grad():
            outputs = self.parsbert(**inputs)
            features = outputs.pooler_output

            # تحلیل معنایی با مدل معلم
            semantic_features = self.semantic_analyzer(features)

            # شبیه‌سازی استخراج اطلاعات معنایی
            intent = self.detect_intent(text)

            # شبیه‌سازی تحلیل احساسات
            sentiment_score = torch.sum(semantic_features).item()
            sentiment = "positive" if sentiment_score > 0.5 else "negative" if sentiment_score < -0.5 else "neutral"

            # شبیه‌سازی استخراج موضوعات
            topics = ["سیاست", "اقتصاد", "فرهنگ"] if len(text) > 50 else ["عمومی"]

            return {
                "intent": intent,
                "sentiment": sentiment,
                "topics": topics,
                "embedding": semantic_features.cpu().numpy().tolist()
            }

    def analyze_literary_style(self, text):
        """تحلیل سبک ادبی متن"""
        if not self.ready:
            return "UNKNOWN"

        # شبیه‌سازی تحلیل سبک ادبی
        inputs = self._preprocess_text(text)
        with torch.no_grad():
            outputs = self.parsbert(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)

            # بر اساس میانگین بردار، سبک را مشخص می‌کنیم (شبیه‌سازی)
            embedding_sum = torch.sum(embedding).item()

            if embedding_sum > 10:
                return "FORMAL"
            elif embedding_sum > 5:
                return "TECHNICAL_PROSE"
            elif embedding_sum > 0:
                return "JOURNALISTIC"
            elif embedding_sum > -5:
                return "COLLOQUIAL"
            else:
                return "INFORMAL"

    def save_model(self, path=CONFIG["teacher_model"]):
        """ذخیره‌سازی مدل معلم"""
        if not self.ready:
            return

        state_dict = {
            "parsbert": self.parsbert.state_dict(),
            "intent_classifier": self.intent_classifier.state_dict() if self.intent_classifier else None,
            "grammar_analyzer": self.grammar_analyzer.state_dict() if self.grammar_analyzer else None,
            "semantic_analyzer": self.semantic_analyzer.state_dict() if self.semantic_analyzer else None
        }
        torch.save(state_dict, path)

    def load_model(self, path=CONFIG["teacher_model"]):
        """بارگذاری مدل معلم از فایل"""
        if not os.path.exists(path):
            return

        try:
            state_dict = torch.load(path)
            if self.parsbert and "parsbert" in state_dict:
                self.parsbert.load_state_dict(state_dict["parsbert"])
            if self.intent_classifier and "intent_classifier" in state_dict:
                self.intent_classifier.load_state_dict(state_dict["intent_classifier"])
            if self.grammar_analyzer and "grammar_analyzer" in state_dict:
                self.grammar_analyzer.load_state_dict(state_dict["grammar_analyzer"])
            if self.semantic_analyzer and "semantic_analyzer" in state_dict:
                self.semantic_analyzer.load_state_dict(state_dict["semantic_analyzer"])
            self.ready = True
        except Exception as e:
            logging.error(f"خطا در بارگذاری مدل معلم: {e}")