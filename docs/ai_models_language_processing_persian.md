# 📚 مستندات ماژول پردازش زبان فارسی در Smart Whale AI

## 📌 مقدمه و هدف
ماژول **Persian Language Processing** یکی از زیرسیستم‌های کلیدی در معماری **Smart Whale AI** است که وظیفه پردازش تخصصی زبان فارسی را با بالاترین دقت و کمترین مصرف منابع بر عهده دارد. این ماژول به عنوان بخشی از `ai/models/language/adaptors/persian/` عمل می‌کند و از معماری ماژولار برای یکپارچگی با سایر بخش‌های سیستم بهره می‌برد.

### 🎯 اهداف اصلی
- **پردازش تخصصی زبان فارسی با بالاترین دقت**
- **مدیریت بهینه منابع پردازشی و حافظه**
- **استفاده از مکانیزم های خودآموزی هوشمند**
- **انتقال تدریجی از وابستگی به مدل معلم به سمت استقلال**
- **کاهش پیچیدگی با استاندارد‌سازی و یکپارچه‌سازی با ماژول‌های دیگر**
- **یکپارچگی کامل با زیرساخت‌های پردازش کوانتومی**

## 📐 ساختار کلی ماژول

```
ai/models/language/adaptors/persian/
├── core/                                # پردازش‌های پایه و بهینه‌سازی اولیه
│   ├── analyzer/                        # تحلیل‌های اولیه متن
│   │   ├── text_analyzer.py             # تحلیل کلی متن
│   │   └── structure_analyzer.py        # تحلیل ساختار جملات
│   ├── processor/                       # پردازشگرهای اصلی فارسی
│   │   ├── text_normalizer.py           # نرمال‌سازی متن فارسی
│   │   ├── quantum_vectorizer.py        # بردارسازی کوانتومی متن فارسی
│   │   └── adaptive_pipeline.py         # پایپلاین پردازش تطبیقی
│   ├── generator/                       # تولیدکننده پاسخ فارسی
│   │   ├── response_generator.py        # تولید پاسخ اصلی
│   │   └── quantum_pipeline.py          # پایپلاین کوانتومی تولید پاسخ
│   └── optimizer/                       # بهینه‌سازهای پردازش
│       ├── quantum_compressor.py        # فشرده‌سازی داده‌های فارسی
│       └── quantum_allocator.py         # تخصیص منابع پردازشی
│
├── language_processors/                 # پردازش‌های تخصصی زبان فارسی
│   ├── analyzer/                        # تحلیل‌های زبانی مختلف
│   ├── contextual/                      # مدیریت و تحلیل زمینه
│   ├── dialects/                        # تشخیص و پردازش گویش‌ها
│   ├── domain/                          # مدیریت دانش دامنه‌ها
│   ├── grammar/                         # تحلیل گرامری متن
│   ├── literature/                      # تحلیل ادبیات فارسی
│   ├── proverbs/                        # مدیریت ضرب‌المثل‌ها
│   └── semantics/                       # تحلیل معنایی عمیق متن
│
├── learning/                           # سیستم‌های یادگیری مختص زبان فارسی
│   ├── trainer_adaptor.py              # اتصال‌دهنده به trainer در ماژول learning
│   ├── validator_adaptor.py            # اتصال‌دهنده به validator در ماژول learning
│   ├── distillation_adaptor.py         # اتصال‌دهنده به distillation در ماژول learning
│   ├── optimizer_adaptor.py            # اتصال‌دهنده به optimizer در ماژول learning
│   └── analytics_adaptor.py            # اتصال‌دهنده به analytics در ماژول learning
│
├── services/                           # سرویس‌های ارتباطی با سایر اجزا
│   ├── kafka_service.py                # ارتباط با کافکا
│   ├── redis_service.py                # ارتباط با ردیس
│   ├── vectordb_service.py             # ارتباط با پایگاه داده برداری
│   └── timescaledb_service.py          # ارتباط با تایم‌اسکیل
│
├── config/                             # تنظیمات اختصاصی زبان فارسی
│   ├── default_config.py               # تنظیمات پیش‌فرض
│   └── persian_config.py  
│
├── self_learning/                       
│
├── smart_model.py                      # مدل دانش‌آموز فارسی
├── teacher.py                          # مدل معلم (در آینده حذف خواهد شد)
└── language_processor.py               # رابط اصلی پردازش زبان فارسی
```

## 🧩 توضیحات کلیدی اجزای اصلی

### 1️⃣ language_processor.py

فایل `language_processor.py` رابط اصلی برای پردازش زبان فارسی است که تمام قابلیت‌های پردازش زبان فارسی را فراهم می‌کند. این کلاس از `BaseLanguageProcessor` ارث‌بری کرده و ارتباط یکپارچه با سایر بخش‌های سیستم را فراهم می‌کند.

```python
# ai/models/language/adaptors/persian/language_processor.py

from ai.models.language.core.base.processor import BaseLanguageProcessor
from ai.models.language.adaptors.persian.language_processors.analyzer import PersianAnalyzer
from ai.models.language.adaptors.persian.language_processors.contextual import PersianContextProcessor
from ai.models.language.adaptors.persian.core.processor import PersianCoreProcessor
from ai.models.language.adaptors.persian.smart_model import PersianSmartModel

class PersianLanguageProcessor(BaseLanguageProcessor):
    """
    رابط اصلی پردازش زبان فارسی که تمام قابلیت‌های پردازش زبان فارسی را در اختیار می‌گذارد
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.model = PersianSmartModel()
        self.analyzer = PersianAnalyzer()
        self.context_processor = PersianContextProcessor()
        self.core_processor = PersianCoreProcessor()
        
    async def process(self, text, context=None, processing_level="normal"):
        """
        پردازش متن فارسی و تولید پاسخ
        
        Args:
            text (str): متن ورودی
            context (dict, optional): زمینه گفتگو
            processing_level (str, optional): سطح پردازش (quick, normal, deep)
            
        Returns:
            dict: پاسخ پردازش‌شده به همراه متادیتا
        """
        # پردازش اولیه با استفاده از core
        processed_text = await self.core_processor.preprocess(text)
        
        # تحلیل متن
        analysis_result = await self.analyzer.analyze(processed_text, processing_level)
        
        # پردازش زمینه
        context_enhanced = await self.context_processor.process(analysis_result, context)
        
        # تولید پاسخ با استفاده از مدل
        response = await self.model.generate(context_enhanced)
        
        return response
```

### 2️⃣ smart_model.py

فایل `smart_model.py` شامل کلاس `PersianSmartModel` است که مدل دانش‌آموز فارسی را مدیریت می‌کند. این مدل با استفاده از موتور خودآموزی پرشین (PersianSelfLearningEngine) به تدریج از مدل معلم مستقل می‌شود.

```python
# ai/models/language/adaptors/persian/smart_model.py

from ai.models.core.base.model import BaseModel
from ai.models.language.adaptors.persian.self_learning.persian_engine import PersianSelfLearningEngine
from ai.models.language.adaptors.persian.teacher import PersianTeacherModel

class PersianSmartModel(BaseModel):
    """
    مدل دانش‌آموز فارسی که به‌تدریج از مدل معلم مستقل می‌شود
    """
    
    def __init__(self, config=None):
        super().__init__("persian-smart-model", "1.0", config)
        self.teacher_model = PersianTeacherModel()
        self.self_learning_engine = PersianSelfLearningEngine(model_id="persian-smart-model")
        self.evolution_phase = 0.0  # میزان استقلال از معلم (0 تا 1)
        
    async def generate(self, context):
        """
        تولید پاسخ بر اساس زمینه مکالمه
        
        با توجه به میزان تکامل مدل (evolution_phase)، از ترکیب پاسخ‌های
        مدل معلم و مدل مستقل استفاده می‌کند
        """
        # دریافت پاسخ از مدل معلم
        teacher_response = await self.teacher_model.generate(context)
        
        # دریافت پاسخ از مدل مستقل (اگر به حد کافی تکامل یافته باشد)
        if self.evolution_phase > 0.1:
            independent_response = await self._generate_independent(context)
            
            # ترکیب پاسخ بر اساس میزان تکامل
            final_response = self._merge_responses(
                teacher_response, 
                independent_response, 
                self.evolution_phase
            )
            
            # ارزیابی نتایج و بهبود میزان تکامل
            new_phase = await self.self_learning_engine.evaluate_and_evolve(
                context,
                teacher_response,
                independent_response,
                self.evolution_phase
            )
            
            self.evolution_phase = new_phase
            return final_response
        
        # اگر مدل هنوز به حد کافی تکامل نیافته، از پاسخ معلم استفاده می‌شود
        return teacher_response
    
    async def _generate_independent(self, context):
        """تولید پاسخ مستقل بدون کمک معلم"""
        # پیاده‌سازی منطق تولید پاسخ مستقل
        pass
    
    def _merge_responses(self, teacher_response, independent_response, evolution_phase):
        """ترکیب پاسخ‌های معلم و مستقل بر اساس میزان تکامل"""
        # پیاده‌سازی منطق ترکیب پاسخ‌ها
        pass
```

### 3️⃣ teacher.py

فایل `teacher.py` شامل کلاس `PersianTeacherModel` است که مدل معلم را مدیریت می‌کند. این مدل بهینه‌شده است تا در مراحل اولیه تکامل مدل دانش‌آموز کمک کند و به تدریج نقش آن کاهش می‌یابد.

```python
# ai/models/language/adaptors/persian/teacher.py

from ai.models.core.base.model import BaseModel

class PersianTeacherModel(BaseModel):
    """
    مدل معلم فارسی که برای آموزش مدل دانش‌آموز استفاده می‌شود
    این مدل در طول زمان و با افزایش استقلال مدل دانش‌آموز، نقش کمتری خواهد داشت
    """
    
    def __init__(self, config=None):
        super().__init__("persian-teacher-model", "2.0", config)
        # بارگذاری مدل آموزش‌دیده
        
    async def generate(self, context):
        """
        تولید پاسخ با استفاده از مدل آموزش‌دیده
        """
        # پیاده‌سازی تولید پاسخ توسط مدل معلم
        response = {
            "text": "پاسخ مدل معلم",
            "metadata": {
                "source": "teacher_model",
                "confidence": 0.95
            }
        }
        return response
```

### 4️⃣ یکپارچگی با مکانیزم Self-Learning

ماژول learning از طریق یک سری `adaptor` با ماژول مرکزی `ai/models/language/learning/` ارتباط برقرار می‌کند و مکانیزم self-learning آن از ماژول متمرکز `ai/models/language/adaptors/persian/self_learning/` استفاده می‌کند.

```python
# ai/models/language/adaptors/persian/learning/trainer_adaptor.py

from ai.models.language.learning.trainer.base_trainer import BaseTrainer
from ai.models.language.adaptors.persian.language_processors.analyzer import PersianAnalyzer

class PersianTrainerAdaptor:
    """
    تطبیق‌دهنده برای ارتباط مدل پرشین با ماژول trainer مرکزی
    """
    
    def __init__(self, model_id="persian-smart-model"):
        self.model_id = model_id
        self.base_trainer = BaseTrainer(model_id)
        self.persian_analyzer = PersianAnalyzer()
    
    async def train(self, data, config=None):
        """
        آموزش مدل با در نظر گرفتن ویژگی‌های خاص زبان فارسی
        """
        # پیش‌پردازش داده‌های فارسی
        processed_data = await self._preprocess_persian_data(data)
        
        # استفاده از ترینر پایه برای آموزش
        return await self.base_trainer.train(processed_data, config)
    
    async def _preprocess_persian_data(self, data):
        """پیش‌پردازش داده‌های فارسی برای آموزش بهتر"""
        # نرمال‌سازی و پیش‌پردازش‌های خاص زبان فارسی
        normalized_data = []
        for item in data:
            if isinstance(item["text"], str):
                # نرمال‌سازی متن فارسی
                analyzed = await self.persian_analyzer.analyze(item["text"])
                item["processed_text"] = analyzed["normalized_text"]
                item["features"] = analyzed["features"]
                normalized_data.append(item)
                
        return normalized_data
```

## 🔄 جریان داده‌ها در ماژول

جریان داده‌ها در ماژول پردازش زبان فارسی به صورت زیر انجام می‌شود:

1. **دریافت ورودی**:
   - تشخیص زبان توسط `language/adaptors/multilingual/language_detector.py` انجام می‌شود.
   - در صورت تشخیص متن فارسی، متن به `language/adaptors/persian/language_processor.py` ارسال می‌شود.

2. **پردازش اولیه**:
   - متن توسط `persian/core/processor/text_normalizer.py` نرمال‌سازی می‌شود.
   - بردارسازی با استفاده از `persian/core/processor/quantum_vectorizer.py` انجام می‌شود.
   - مدیریت اولیه و انتخاب مسیر پردازش مناسب توسط `persian/core/processor/adaptive_pipeline.py` انجام می‌شود.

3. **پردازش تخصصی فارسی**:
   - متن نرمال‌شده به پوشه‌ی `persian/language_processors/` منتقل می‌شود.
   - تحلیل‌های تخصصی (نحوی، معنایی، زمینه‌ای، گرامری و...) روی آن انجام می‌گیرد.

4. **تولید پاسخ**:
   - با استفاده از `persian/smart_model.py`، پاسخ نهایی تولید می‌شود.
   - پاسخ تولیدشده توسط `persian/core/generator/response_generator.py` پردازش نهایی می‌شود.

5. **یادگیری و بهبود مستمر**:
   - تعامل کاربر و پاسخ‌های مدل توسط `persian/learning/` جمع‌آوری می‌شود.
   - داده‌ها برای آموزش مدل و بهبود عملکرد استفاده می‌شوند.
   - مکانیزم خودآموزی `persian/smart_model.py` به‌تدریج مدل را بهبود می‌بخشد.

## 🔌 یکپارچگی با سایر ماژول‌ها

ماژول پردازش زبان فارسی با ماژول‌های زیر ارتباط دارد:

### 1. یکپارچگی با ماژول Federation

فایل‌های `persian/learning/` با `ai/models/language/federation/` ارتباط برقرار می‌کنند تا امکان اشتراک دانش بین مدل‌های مختلف را فراهم کنند:

```python
# نمونه استفاده از Federation در پرشین
from ai.models.language.federation.knowledge_sharing.knowledge_manager import KnowledgeManager

knowledge_manager = KnowledgeManager()
shared_knowledge = knowledge_manager.get_shared_knowledge("grammar_correction")
```

### 2. یکپارچگی با ماژول Core

فایل‌های `persian/core/` با `ai/models/core/` ارتباط برقرار می‌کنند تا از قابلیت‌های پردازشی و حافظه کوانتومی بهره‌مند شوند:

```python
# نمونه استفاده از quantum memory در پرشین
from ai.core.memory.quantum_memory import QuantumMemory

quantum_memory = QuantumMemory()
compressed_text = quantum_memory.compress("متن فارسی برای فشرده‌سازی")
```

### 3. یکپارچگی با زیرساخت‌ها

فایل‌های `persian/services/` با `infrastructure/` ارتباط برقرار می‌کنند:

```python
# نمونه استفاده از Redis در پرشین
from infrastructure.redis.service.cache_service import CacheService

cache_service = CacheService()
await cache_service.connect()
await cache_service.set('persian_response', {'text': 'پاسخ'}, ttl=3600)
```

### 4. یکپارچگی با ماژول Learning

کلاس‌های adaptor در `persian/learning/` با `ai/models/language/learning/` ارتباط برقرار می‌کنند:

```python
# نمونه استفاده از learning در پرشین
from ai.models.language.learning.trainer.base_trainer import BaseTrainer

trainer = BaseTrainer("persian-model")
await trainer.train(persian_data)
```

## 🔮 مکانیزم‌های پیشرفته

### 1. پردازش کوانتومی و فشرده‌سازی داده‌ها

ماژول پرشین از فناوری‌های پردازش کوانتومی برای بهینه‌سازی عملکرد استفاده می‌کند:

```python
# ai/models/language/adaptors/persian/core/processor/quantum_vectorizer.py

from ai.core.memory.quantum_memory import QuantumMemory

class QuantumVectorizer:
    """
    تبدیل متن فارسی به بردارهای کوانتومی برای بهینه‌سازی پردازش
    """
    
    def __init__(self):
        self.quantum_memory = QuantumMemory()
        
    def transform(self, text):
        """تبدیل متن به بردار کوانتومی"""
        # پیاده‌سازی بردارسازی کوانتومی
        return self.quantum_memory.vectorize(text)
```

### 2. مکانیزم خودآموزی (Self-Learning)

مکانیزم خودآموزی مدل پرشین از `ai/models/language/adaptors/persian/self_learning/persian_engine.py` استفاده می‌کند:

```python
# ai/models/language/adaptors/persian/self_learning/persian_engine.py

from ai.models.language.learning.self_learning.base.learning_engine import SelfLearningEngine
from ai.models.language.adaptors.persian.self_learning.need_detection.persian_gap_analyzer import PersianGapAnalyzer
from ai.models.language.adaptors.persian.self_learning.strategy.persian_cultural_strategy import PersianCulturalStrategy

class PersianSelfLearningEngine(SelfLearningEngine):
    """
    موتور خودآموزی اختصاصی زبان فارسی
    """
    
    def __init__(self, model_id):
        super().__init__(model_id)
        
        # جایگزینی اجزای پایه با نسخه‌های اختصاصی فارسی
        self.gap_analyzer = PersianGapAnalyzer(model_id, self.lifecycle_manager)
        
        # اضافه کردن استراتژی‌های خاص فارسی
        self.strategy_factory.register_strategy("CULTURAL", PersianCulturalStrategy())
    
    async def evaluate_and_evolve(self, context, teacher_response, independent_response, current_phase):
        """
        ارزیابی پاسخ‌های مدل معلم و مدل مستقل و بهبود مرحله تکاملی
        """
        # پیاده‌سازی ارزیابی و تکامل مدل
        performance_delta = self._calculate_performance_delta(
            teacher_response, independent_response, context
        )
        
        # تنظیم مرحله تکامل
        if performance_delta > 0:
            # افزایش تدریجی استقلال
            new_phase = min(1.0, current_phase + (0.01 * performance_delta))
        else:
            # کاهش استقلال در صورت عملکرد ضعیف
            new_phase = max(0.1, current_phase - (0.005 * abs(performance_delta)))
            
        return new_phase
```

## ⚙️ تنظیمات و پیکربندی

تنظیمات مدل پرشین در پوشه `config/` قرار دارد:

```python
# ai/models/language/adaptors/persian/config/persian_config.py

PERSIAN_CONFIG = {
    # تنظیمات عمومی
    "model_name": "persian-smart-model",
    "version": "1.0.0",
    
    # تنظیمات پردازش
    "processing": {
        "default_level": "normal",  # سطح پردازش پیش‌فرض (quick, normal, deep)
        "normalization": {
            "remove_diacritics": True,  # حذف اعراب
            "fix_spacing": True,  # اصلاح فاصله‌گذاری
            "use_persian_numbers": True,  # تبدیل اعداد انگلیسی به فارسی
        },
    },
    
    # تنظیمات سلف‌لرنینگ
    "self_learning": {
        "initial_phase": 0.1,  # مرحله تکامل اولیه
        "learning_rate": 0.01,  # نرخ یادگیری
        "max_concurrent_requests": 5,  # حداکثر درخواست‌های همزمان
    },
    
    # تنظیمات کش
    "cache": {
        "ttl": 3600,  # زمان نگهداری در کش (ثانیه)
        "max_size": 10000,  # حداکثر تعداد آیتم‌های کش
    },
}
```




## 🎯 خلاصه و نتیجه‌گیری

ماژول پردازش زبان فارسی در Smart Whale AI با رویکرد یکپارچه‌سازی کامل با سایر ماژول‌ها طراحی شده است. این ماژول:

1. **از سرویس‌های مشترک استفاده می‌کند** تا از تکرار کد جلوگیری شود.
2. **بدون افزونگی با ماژول‌های اصلی ارتباط برقرار می‌کند** و در عین حال قابلیت‌های تخصصی زبان فارسی را ارائه می‌دهد.
3. **از مکانیزم خودآموزی مرکزی بهره می‌برد** و آن را با نیازهای خاص زبان فارسی تطبیق می‌دهد.
4. **از فناوری‌های پیشرفته پردازش کوانتومی استفاده می‌کند** تا کارایی بالاتر با مصرف منابع کمتر را فراهم آورد.
5. **یک جریان داده یکپارچه را طراحی می‌کند** که از ورودی تا خروجی، بهترین کارایی را با کمترین سربار ارائه می‌دهد.

این ساختار استاندارد می‌تواند برای سایر زبان‌ها نیز استفاده شود و بستر مناسبی برای توسعه و بهبود مستمر ماژول پردازش زبان فارسی فراهم می‌کند.



# مستندات بخش Self-Learning ماژول Persian

## 📌 معرفی

بخش `self_learning` در ماژول `persian` یک سیستم خودآموزی پیشرفته و تخصصی برای زبان فارسی است که با هدف یادگیری خودکار و تکامل تدریجی مدل از وابستگی به معلم به سمت استقلال طراحی شده است. این بخش برپایه معماری مرکزی Self-Learning در `ai/models/language/learning/self_learning/` بنا شده، اما با ویژگی‌های تخصصی برای زبان فارسی گسترش یافته است.

## 📁 ساختار بخش Self-Learning

```
ai/models/language/adaptors/persian/self_learning/
├── persian_engine.py               # موتور اصلی خودآموزی مختص فارسی
│
├── need_detection/
│   ├── persian_gap_analyzer.py      # تحلیل شکاف‌های خاص زبان فارسی
│   ├── persian_trend_detector.py    # تشخیص روندهای فارسی‌زبان
│   └── persian_feedback_analyzer.py # تحلیل بازخوردهای فارسی‌زبان
│
├── acquisition/
│   ├── persian_source_selector.py   # انتخاب منابع فارسی‌زبان
│   └── persian_query_generator.py   # تولید کوئری‌های فارسی
│
├── processing/
│   ├── persian_text_cleaner.py      # تمیزسازی متون فارسی
│   ├── normalization_handler.py     # مدیریت نرمال‌سازی متون فارسی
│   └── persian_knowledge_mapper.py  # نگاشت دانش به ساختارهای فارسی
│
├── strategy/
│   ├── persian_cultural_strategy.py # استراتژی مبتنی بر فرهنگ فارسی
│   └── dialect_aware_strategy.py    # استراتژی آگاه از لهجه‌های فارسی
│
└── config/
    └── persian_config.py            # تنظیمات خاص خودآموزی زبان فارسی
```

## 🔍 شرح اجزای اصلی

### 1️⃣ persian_engine.py
موتور اصلی خودآموزی مختص زبان فارسی که از کلاس پایه `SelfLearningEngine` ارث‌بری می‌کند و رفتارهای خاص زبان فارسی را پیاده‌سازی می‌کند. این موتور مسئول هماهنگی بین تمام اجزای سیستم خودآموزی مختص فارسی است و چرخه‌های یادگیری را مدیریت می‌کند.

### 2️⃣ need_detection/
این بخش مسئول شناسایی نیازهای یادگیری خاص زبان فارسی است:
- **persian_gap_analyzer.py**: تشخیص شکاف‌های دانشی در واژگان، گرامر و ساختارهای خاص فارسی
- **persian_trend_detector.py**: شناسایی روندهای جدید در زبان فارسی، اصطلاحات رایج و مفاهیم نوظهور
- **persian_feedback_analyzer.py**: تحلیل بازخوردهای کاربران فارسی‌زبان برای شناسایی حوزه‌های نیازمند بهبود

### 3️⃣ acquisition/
این بخش مسئول جمع‌آوری و مدیریت داده‌های مورد نیاز برای یادگیری است:
- **persian_source_selector.py**: انتخاب هوشمند منابع معتبر فارسی برای یادگیری
- **persian_query_generator.py**: تولید پرس‌وجوهای هوشمند برای جمع‌آوری داده‌های آموزشی فارسی

### 4️⃣ processing/
این بخش مسئول پردازش و آماده‌سازی داده‌های فارسی برای یادگیری است:
- **persian_text_cleaner.py**: پاکسازی متون فارسی از خطاها و ناهنجاری‌ها
- **normalization_handler.py**: نرمال‌سازی تخصصی متون فارسی (یکسان‌سازی "ی"ها، نقطه‌گذاری، اعراب و...)
- **persian_knowledge_mapper.py**: تبدیل دانش استخراج‌شده به ساختارهای مناسب برای یادگیری مدل فارسی

### 5️⃣ strategy/
این بخش استراتژی‌های تخصصی یادگیری برای زبان فارسی را ارائه می‌دهد:
- **persian_cultural_strategy.py**: استراتژی یادگیری مبتنی بر ویژگی‌های فرهنگی و بافتاری خاص زبان فارسی
- **dialect_aware_strategy.py**: استراتژی یادگیری آگاه از لهجه‌های مختلف فارسی (تهرانی، اصفهانی، مشهدی و...)

## 🔄 چرخه کاری Self-Learning فارسی

1. **شناسایی نیاز**: تشخیص شکاف‌های دانشی با استفاده از `persian_gap_analyzer.py` و تحلیل بازخوردها
2. **اولویت‌بندی**: استفاده از استراتژی‌های فرهنگی و زبانی برای اولویت‌بندی نیازهای یادگیری
3. **جمع‌آوری داده**: استخراج داده از منابع معتبر فارسی با استفاده از `persian_source_selector.py`
4. **پردازش**: نرمال‌سازی و تمیزسازی داده‌ها با استفاده از ابزارهای تخصصی فارسی
5. **یادگیری**: اعمال دانش جدید در مدل با توجه به مرحله تکاملی آن
6. **ارزیابی**: سنجش میزان پیشرفت و تنظیم استراتژی‌های یادگیری بعدی

## 🎯 مزایای خاص Self-Learning فارسی

- **درک فرهنگی**: توانایی درک ضرب‌المثل‌ها، کنایه‌ها و استعاره‌های خاص فرهنگ ایرانی
- **تطبیق لهجه‌ای**: توانایی تشخیص و تطبیق با لهجه‌های مختلف فارسی
- **هوشمندی زمینه‌ای**: درک بهتر زمینه در مکالمات فارسی با توجه به تعارفات و ادبیات خاص فارسی
- **استقلال تدریجی**: حرکت موفقیت‌آمیز از وابستگی به مدل معلم به سمت مدل کاملاً مستقل متناسب با زبان فارسی

این سیستم خودآموزی به مدل فارسی امکان می‌دهد تا به‌طور مستمر، با حداقل ورودی انسانی و با کمترین مصرف منابع، عملکرد خود را بهبود بخشد و به‌تدریج به مدلی کاملاً مستقل و با دقت بالا تبدیل شود.