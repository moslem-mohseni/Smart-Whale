from ..services.redis_service import RedisClient
from ..services.clickhouse_service import ClickHouseClient
from ..services.kafka_service import KafkaProducer, KafkaConsumer
from hazm import Normalizer, POSTagger, Lemmatizer


class GrammarAnalyzer:
    """
    پردازش گرامری و تحلیل اشتباهات دستوری در زبان فارسی.
    """

    def __init__(self):
        self.redis = RedisClient()
        self.clickhouse = ClickHouseClient()
        self.kafka_producer = KafkaProducer(topic="grammar_updates")
        self.kafka_consumer = KafkaConsumer(topic="grammar_updates")

        # ابزارهای پردازش متن فارسی
        self.normalizer = Normalizer()
        self.pos_tagger = POSTagger(model="resources/postagger.model")
        self.lemmatizer = Lemmatizer()

    def analyze_grammar(self, text):
        """
        تحلیل گرامری و شناسایی اشتباهات در متن.

        :param text: متن ورودی
        :return: لیست اشتباهات گرامری
        """
        text = self.normalizer.normalize(text)
        words = text.split()
        tagged_words = self.pos_tagger.tag(words)

        grammar_errors = []
        for word, tag in tagged_words:
            if tag in ["V", "N", "ADJ"]:  # فقط افعال، اسم‌ها و صفات بررسی شوند
                lemma = self.lemmatizer.lemmatize(word)
                if word != lemma:  # اگر شکل واژه اصلاح‌شده با ورودی متفاوت باشد، احتمال اشتباه وجود دارد
                    grammar_errors.append({"word": word, "suggested": lemma, "error_type": "Misconjugation"})

        return grammar_errors

    def correct_text(self, text):
        """
        اصلاح گرامری متن و جایگزینی اشتباهات با کلمات صحیح.

        :param text: متن ورودی
        :return: متن اصلاح‌شده
        """
        errors = self.analyze_grammar(text)
        for error in errors:
            text = text.replace(error["word"], error["suggested"])
        return text

    def save_correction(self, original_text, corrected_text):
        """
        ذخیره‌ی تغییرات گرامری در پایگاه دانش.

        :param original_text: متن اولیه‌ی دارای اشتباه
        :param corrected_text: متن اصلاح‌شده
        """
        self.redis.set(f"grammar:{original_text}", corrected_text)
        self.clickhouse.insert("grammar_corrections", {"original": original_text, "corrected": corrected_text})
        self.kafka_producer.send({"original": original_text, "corrected": corrected_text})

    def get_correction(self, text):
        """
        دریافت نسخه‌ی اصلاح‌شده‌ی یک متن از پایگاه داده.

        :param text: متن ورودی
        :return: متن اصلاح‌شده (در صورت موجود بودن)
        """
        return self.redis.get(f"grammar:{text}") or text  # اگر موجود نبود، همان متن را بازمی‌گرداند


# =========================== TEST ===========================
if __name__ == "__main__":
    grammar = GrammarAnalyzer()

    test_text = "او رفتن به مدرسه"
    print("📌 متن اصلی:", test_text)

    errors = grammar.analyze_grammar(test_text)
    print("📌 اشتباهات گرامری:", errors)

    corrected_text = grammar.correct_text(test_text)
    print("📌 متن اصلاح‌شده:", corrected_text)

    grammar.save_correction(test_text, corrected_text)
    print("📌 بازیابی متن اصلاح‌شده از پایگاه داده:", grammar.get_correction(test_text))
