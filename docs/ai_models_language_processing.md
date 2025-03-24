# مستندات ماژول Language Processing

## 🎯 هدف ماژول
ماژول `language_processing` یک موتور قدرتمند برای پردازش چندزبانه است که با رویکردی بهینه و هوشمند طراحی شده است. این ماژول از معماری ماژولار برای پشتیبانی از زبان‌های مختلف استفاده می‌کند، در حالی که منابع سیستم را به صورت بهینه مدیریت می‌کند.

## 📂 ساختار ماژول
```
language/
├── core/                           # هسته مشترک و پایه
│   ├── base_model.py              # کلاس پایه مدل‌های زبانی
│   ├── resource_manager.py        # مدیریت منابع سیستم
│   ├── mixed_language_handler.py  # مدیریت متون چندزبانه
│   └── evolution_manager.py       # مدیریت تکامل مدل‌ها
│
├── persian/                       # ماژول زبان فارسی
│   └── [مستندات جداگانه]
│
├── english/                       # ماژول زبان انگلیسی
│   └── [مستندات جداگانه]
│
└── interfaces/                    # رابط‌های ارتباطی
    ├── kafka_interface.py         # رابط با کافکا
    ├── telegram_interface.py      # رابط با تلگرام
    └── api_interface.py          # رابط‌های API
```

## 🛠 بخش Core

### BaseModel (`base_model.py`)
کلاس پایه برای تمام مدل‌های زبانی که اصول و عملکردهای مشترک را تعریف می‌کند.

```python
class LanguageModel(ABC):
    def __init__(self, model_config: dict):
        self.model_config = model_config
        self.evolution_phase = 0.0  # میزان تکامل مدل (0 تا 1)
        self.resource_monitor = ResourceMonitor()

    @abstractmethod
    async def process_text(self, text: str) -> dict:
        """پردازش متن و برگرداندن نتیجه"""
        pass

    @abstractmethod
    async def train(self, data: List[str]) -> None:
        """آموزش مدل با داده‌های جدید"""
        pass

    def get_resource_usage(self) -> dict:
        """گزارش مصرف منابع"""
        return self.resource_monitor.get_stats()
```

### ResourceManager (`resource_manager.py`)
مدیریت هوشمند منابع سیستم برای بهینه‌سازی مصرف حافظه و CPU.

```python
class ResourceManager:
    def __init__(self):
        self.memory_threshold = 0.8  # حداکثر مصرف مجاز حافظه
        self.cpu_threshold = 0.9     # حداکثر مصرف مجاز CPU
        self.active_models = {}      # مدل‌های فعال در حافظه

    async def optimize_resource_usage(self):
        """بهینه‌سازی مصرف منابع"""
        if self.get_memory_usage() > self.memory_threshold:
            await self._free_inactive_models()
        
        if self.get_cpu_usage() > self.cpu_threshold:
            await self._throttle_processing()

    async def load_model(self, language: str) -> bool:
        """بارگذاری هوشمند مدل زبانی"""
        if not self._has_sufficient_resources():
            await self.optimize_resource_usage()
        
        return await self._load_model_into_memory(language)
```

### MixedLanguageHandler (`mixed_language_handler.py`)
مدیریت متون چندزبانه و تشخیص هوشمند زبان‌ها.

```python
class MixedLanguageHandler:
    def __init__(self):
        self.language_detectors = {}
        self.processing_cache = LRUCache(maxsize=1000)

    async def process_mixed_text(self, text: str) -> List[dict]:
        """پردازش متن چندزبانه"""
        # بررسی کش
        cache_key = self._generate_cache_key(text)
        if cached_result := self.processing_cache.get(cache_key):
            return cached_result

        # تقسیم متن به بخش‌های تک‌زبانه
        segments = await self._segment_by_language(text)
        
        # پردازش موازی هر بخش
        results = await asyncio.gather(
            *[self._process_segment(segment) for segment in segments]
        )

        # ذخیره در کش و برگرداندن نتیجه
        self.processing_cache[cache_key] = results
        return results

    async def _segment_by_language(self, text: str) -> List[dict]:
        """تقسیم متن به بخش‌های مجزا بر اساس زبان"""
        segments = []
        current_segment = ""
        current_language = None

        for char in text:
            detected_language = self._detect_char_language(char)
            
            if detected_language != current_language and current_segment:
                segments.append({
                    'text': current_segment,
                    'language': current_language
                })
                current_segment = ""
            
            current_segment += char
            current_language = detected_language

        if current_segment:
            segments.append({
                'text': current_segment,
                'language': current_language
            })

        return segments
```

### EvolutionManager (`evolution_manager.py`)
مدیریت تکامل تدریجی مدل‌ها از وابستگی به مدل‌های پایه به سمت استقلال.

```python
class EvolutionManager:
    def __init__(self):
        self.evolution_rates = {}    # نرخ تکامل هر مدل
        self.performance_metrics = {} # معیارهای عملکرد

    async def track_evolution(self, model_id: str, 
                            base_output: dict,
                            evolved_output: dict) -> float:
        """محاسبه میزان تکامل مدل"""
        performance_delta = self._calculate_performance_delta(
            base_output, evolved_output
        )
        
        current_rate = self.evolution_rates.get(model_id, 0.0)
        new_rate = self._adjust_evolution_rate(
            current_rate, performance_delta
        )
        
        self.evolution_rates[model_id] = new_rate
        return new_rate

    def _adjust_evolution_rate(self, 
                             current_rate: float,
                             performance_delta: float) -> float:
        """تنظیم نرخ تکامل بر اساس عملکرد"""
        if performance_delta > 0:
            # افزایش تدریجی وزن مدل مستقل
            return min(1.0, current_rate + (0.1 * performance_delta))
        else:
            # کاهش سریع در صورت افت عملکرد
            return max(0.0, current_rate - (0.2 * abs(performance_delta)))
```

## 🔌 بخش Interfaces

### KafkaInterface (`kafka_interface.py`)
رابط ارتباطی با Kafka برای دریافت و ارسال پیام‌ها.

```python
class KafkaInterface:
    def __init__(self, config: dict):
        self.producer = KafkaProducer(**config)
        self.consumer = KafkaConsumer(**config)
        self.message_handlers = {}

    async def publish_result(self, 
                           topic: str,
                           result: dict,
                           metadata: dict = None) -> None:
        """انتشار نتیجه پردازش روی کافکا"""
        message = {
            'result': result,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        await self.producer.send(
            topic=topic,
            value=json.dumps(message).encode('utf-8')
        )

    async def start_listening(self, 
                            topics: List[str],
                            group_id: str) -> None:
        """شروع گوش دادن به تاپیک‌های کافکا"""
        self.consumer.subscribe(topics)
        
        async for message in self.consumer:
            handler = self.message_handlers.get(message.topic)
            if handler:
                await handler(message.value)
```

### TelegramInterface (`telegram_interface.py`)
رابط ارتباطی با بات تلگرام.

```python
class TelegramInterface:
    def __init__(self, config: dict):
        self.kafka_interface = KafkaInterface(config['kafka'])
        self.response_cache = TTLCache(
            maxsize=1000,
            ttl=300  # 5 minutes
        )

    async def handle_message(self, 
                           message: dict,
                           chat_id: str) -> None:
        """مدیریت پیام‌های دریافتی از تلگرام"""
        # انتشار پیام روی کافکا
        await self.kafka_interface.publish_result(
            topic='telegram_messages',
            result={
                'text': message['text'],
                'chat_id': chat_id
            }
        )

        # ثبت chat_id در کش برای پاسخ
        self.response_cache[chat_id] = asyncio.Future()
        
        # منتظر پاسخ می‌مانیم
        response = await self.response_cache[chat_id]
        await self._send_telegram_response(chat_id, response)
```

### APIInterface (`api_interface.py`)
رابط API برای ارتباط با سایر سرویس‌ها.

```python
class APIInterface:
    def __init__(self):
        self.rate_limiter = RateLimiter(
            max_requests=100,
            time_window=60  # 1 minute
        )
        self.request_validator = RequestValidator()

    async def handle_request(self,
                           request: dict,
                           client_id: str) -> dict:
        """مدیریت درخواست‌های API"""
        # بررسی محدودیت تعداد درخواست
        if not await self.rate_limiter.check_limit(client_id):
            raise RateLimitExceeded()

        # اعتبارسنجی درخواست
        if not self.request_validator.validate(request):
            raise InvalidRequest()

        # پردازش درخواست
        processor = self._get_processor(request['type'])
        result = await processor.process(request['data'])

        return {
            'status': 'success',
            'result': result,
            'request_id': request.get('request_id')
        }
```

## 🔄 جریان کار سیستم

1. **دریافت ورودی:**
   - از طریق API
   - از طریق Kafka
   - از طریق بات تلگرام

2. **تشخیص زبان و تقسیم‌بندی:**
   - تشخیص زبان‌های موجود در متن
   - تقسیم متن به بخش‌های تک‌زبانه
   - ارسال هر بخش به مدل مربوطه

3. **پردازش موازی:**
   - پردازش همزمان بخش‌های مختلف
   - مدیریت بهینه منابع
   - کش‌گذاری نتایج

4. **ترکیب نتایج:**
   - جمع‌آوری نتایج پردازش‌های موازی
   - مرتب‌سازی بر اساس ترتیب اصلی
   - اعمال پردازش‌های نهایی

5. **ارسال پاسخ:**
   - انتشار نتیجه روی Kafka
   - ارسال پاسخ به API
   - ارسال پیام در تلگرام

## 🔍 مانیتورینگ و مدیریت خطا

سیستم به طور مداوم موارد زیر را پایش می‌کند:
- مصرف منابع (CPU, RAM)
- زمان پاسخ‌دهی
- نرخ خطا
- وضعیت کش
- عملکرد مدل‌ها

در صورت بروز خطا:
1. ثبت خطا در سیستم لاگینگ
2. اطلاع‌رسانی به سیستم مانیتورینگ
3. اجرای مکانیزم‌های بازیابی
4. بازگشت به حالت پایدار

## 🎯 اهداف آینده

1. **بهبود تشخیص زبان:**
   - افزودن پشتیبانی از زبان‌های جدید
   - بهبود دقت در متون مخلوط
   - کاهش زمان تشخیص

2. **بهینه‌سازی منابع:**
   - پیاده‌سازی مکانیزم‌های پیشرفته‌تر کش
   - بهبود مدیریت حافظه
   - کاهش زمان پردازش

3. **توسعه رابط‌ها:**
   - افزودن پروتکل‌های جدید
   - بهبود مدیریت خطا
   - افزایش امنیت