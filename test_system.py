import spacy


def test_language_models():
    """تست مدل‌های زبانی نصب شده"""
    print("Testing language models...")

    # تست مدل انگلیسی
    print("\nTesting English model:")
    try:
        nlp_en = spacy.load('en_core_web_lg')
        english_text = "Artificial Intelligence is transforming the world of technology"
        doc_en = nlp_en(english_text)
        print("English model loaded successfully")
        print(f"Tokens: {[token.text for token in doc_en]}")
        print(f"Entities: {[(ent.text, ent.label_) for ent in doc_en.ents]}")
    except Exception as e:
        print(f"Error loading English model: {e}")

    # تست مدل چندزبانه
    print("\nTesting multilingual model:")
    try:
        nlp_multi = spacy.load('xx_ent_wiki_sm')
        persian_text = "هوش مصنوعی در حال تغییر دنیای فناوری است"
        doc_fa = nlp_multi(persian_text)
        print("Multilingual model loaded successfully")
        print(f"Tokens: {[token.text for token in doc_fa]}")
        print(f"Entities: {[(ent.text, ent.label_) for ent in doc_fa.ents]}")
    except Exception as e:
        print(f"Error loading multilingual model: {e}")


if __name__ == "__main__":
    test_language_models()