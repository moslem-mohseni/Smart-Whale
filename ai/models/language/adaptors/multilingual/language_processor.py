import importlib
from config import CONFIG


class LanguageProcessor:
    def __init__(self, language="multilingual"):
        self.language = language
        self.smart_model = self._load_smart_model()
        self.teacher = self._load_teacher_model() if CONFIG["use_teacher_model"] else None

        self.tokenizer = importlib.import_module("tokenization").Tokenizer(language)
        self.normalizer = importlib.import_module("normalization").Normalizer(language)
        self.transliterator = importlib.import_module("transliteration").Transliterator(language)
        self.sentiment_analyzer = importlib.import_module("sentiment_analysis").SentimentAnalyzer(language)
        self.grammar_processor = importlib.import_module("grammar").GrammarProcessor(language)
        self.semantic_processor = importlib.import_module("semantics").SemanticProcessor(language)

    def _load_smart_model(self):
        return importlib.import_module("smart_model").SmartModel()

    def _load_teacher_model(self):
        try:
            return importlib.import_module("teacher").TeacherModel()
        except ModuleNotFoundError:
            return None

    def process_text(self, text):
        """
        اجرای تمامی مراحل پردازش متن
        """
        text = self.normalizer.process(text)
        tokens = self.tokenizer.tokenize(text)
        semantics = self.semantic_processor.analyze_semantics(text)
        sentiment = self.sentiment_analyzer.analyze_sentiment(text)
        grammar = self.grammar_processor.correct_grammar(text)

        return {
            "normalized_text": text,
            "tokens": tokens,
            "semantics": semantics,
            "sentiment": sentiment,
            "grammar_correction": grammar,
        }
