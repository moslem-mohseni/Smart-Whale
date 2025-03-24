from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TeacherModel:
    def __init__(self):
        self.model_name = "bert-base-multilingual-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def tokenize(self, text):
        """
        توکن‌سازی متن با استفاده از مدل معلم `mBERT`
        """
        return self.tokenizer.tokenize(text)

    def normalize(self, text):
        """
        نرمال‌سازی متن با پردازش مدل معلم
        """
        tokens = self.tokenizer.tokenize(text)
        return " ".join(tokens)

    def transliterate(self, text):
        """
        تبدیل نویسه‌های متن بر اساس مدل معلم
        """
        return text  # در این نسخه، `mBERT` مستقیماً از تبدیل نویسه‌ها پشتیبانی نمی‌کند

    def analyze_sentiment(self, text):
        """
        تحلیل احساسات متن با استفاده از مدل معلم `mBERT`
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.logits.detach().numpy()

    def analyze_semantics(self, text):
        """
        پردازش معنایی متن با استفاده از `mBERT`
        """
        return self.tokenizer.encode(text)