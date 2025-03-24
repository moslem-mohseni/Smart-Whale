# 📌 مستندات ماژول `interfaces/` در Balance

## 📂 ساختار ماژول
```
balance/
├── interfaces/                     # رابط‌های ارتباطی
│   ├── model/                      # رابط با مدل‌ها
│   │   ├── model_interface.py      # رابط اصلی مدل‌ها
│   │   ├── request_handler.py      # مدیریت درخواست‌ها
│   │   └── response_handler.py     # مدیریت پاسخ‌ها
│   │
│   ├── data/                       # رابط با داده‌ها
│   │   ├── data_interface.py       # رابط اصلی داده‌ها
│   │   ├── stream_handler.py       # مدیریت جریان داده
│   │   └── sync_handler.py         # مدیریت همگام‌سازی
│   │
│   └── external/                   # رابط‌های خارجی
│       ├── api_interface.py        # رابط API
│       ├── kafka_interface.py      # رابط Kafka
│       └── metrics_interface.py    # رابط متریک‌ها
```

---

## 📌 جزئیات کلاس‌ها و متدهای ماژول `interfaces/`

### **1️⃣ model/** - مدیریت ارتباط با مدل‌ها
#### 🔹 `model_interface.py`
```python
class ModelInterface(ABC):
    def send_request(self, model_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
    def receive_response(self, model_id: str, response_data: Dict[str, Any]) -> None:
```
#### 🔹 `request_handler.py`
```python
class RequestHandler(ABC):
    def validate_request(self, request_data: Dict[str, Any]) -> bool:
    def preprocess_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
```
#### 🔹 `response_handler.py`
```python
class ResponseHandler(ABC):
    def format_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
    def log_response(self, model_id: str, response_data: Dict[str, Any]) -> None:
```

---

### **2️⃣ data/** - مدیریت ارتباط با داده‌ها
#### 🔹 `data_interface.py`
```python
class DataInterface(ABC):
    def fetch_data(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
    def store_data(self, data: Dict[str, Any]) -> None:
```
#### 🔹 `stream_handler.py`
```python
class StreamHandler(ABC):
    def open_stream(self, query_params: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    def close_stream(self) -> None:
```
#### 🔹 `sync_handler.py`
```python
class SyncHandler(ABC):
    def sync_data(self, source: str, destination: str, data: Dict[str, Any]) -> None:
    def verify_sync(self, source: str, destination: str) -> bool:
```

---

### **3️⃣ external/** - ارتباطات خارجی
#### 🔹 `api_interface.py`
```python
class APIInterface(ABC):
    def send_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    def receive_response(self, response_data: Dict[str, Any]) -> None:
```
#### 🔹 `kafka_interface.py`
```python
class KafkaInterface(ABC):
    def publish_message(self, topic: str, message: Dict[str, Any]) -> None:
    def consume_message(self, topic: str) -> Dict[str, Any]:
```
#### 🔹 `metrics_interface.py`
```python
class MetricsInterface(ABC):
    def collect_metrics(self) -> Dict[str, Any]:
    def report_metrics(self, metrics_data: Dict[str, Any]) -> None:
```

---

✅ **تمامی کلاس‌های ماژول `interfaces/` به صورت کامل پیاده‌سازی و مستند شدند.** 🎉





# 📌 مستندات ماژول `monitoring/` در Balance

## 📂 ساختار ماژول
```
balance/
└── monitoring/                     # پایش و گزارش‌گیری
    ├── metrics/                    # متریک‌های سیستم
    │   ├── performance_metrics.py  # متریک‌های کارایی
    │   ├── quality_metrics.py      # متریک‌های کیفیت
    │   └── resource_metrics.py     # متریک‌های منابع
    │
    ├── alerts/                     # سیستم هشدار
    │   ├── alert_detector.py       # تشخیص هشدارها
    │   ├── alert_classifier.py     # دسته‌بندی هشدارها
    │   └── notification_manager.py # مدیریت اطلاع‌رسانی
    │
    └── reporting/                  # گزارش‌دهی
        ├── report_generator.py     # تولید گزارش
        ├── trend_analyzer.py       # تحلیل روندها
        └── dashboard_manager.py    # مدیریت داشبورد
```

---

## 📌 جزئیات کلاس‌ها و متدهای ماژول `monitoring/`

### **1️⃣ metrics/** - متریک‌های سیستم
#### 🔹 `performance_metrics.py`
```python
class PerformanceMetrics(ABC):
    def collect_performance_data(self) -> Dict[str, Any]:
    def analyze_performance_trends(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
```
#### 🔹 `quality_metrics.py`
```python
class QualityMetrics(ABC):
    def evaluate_data_quality(self) -> Dict[str, Any]:
    def monitor_quality_trends(self, quality_data: Dict[str, Any]) -> Dict[str, Any]:
```
#### 🔹 `resource_metrics.py`
```python
class ResourceMetrics(ABC):
    def track_resource_usage(self) -> Dict[str, Any]:
    def analyze_resource_trends(self, resource_data: Dict[str, Any]) -> Dict[str, Any]:
```

---

### **2️⃣ alerts/** - سیستم هشدار
#### 🔹 `alert_detector.py`
```python
class AlertDetector(ABC):
    def detect_anomalies(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
    def trigger_alerts(self, alert_data: Dict[str, Any]) -> None:
```
#### 🔹 `alert_classifier.py`
```python
class AlertClassifier(ABC):
    def classify_alerts(self, alert_data: Dict[str, Any]) -> str:
    def log_alerts(self, alert_data: Dict[str, Any]) -> None:
```
#### 🔹 `notification_manager.py`
```python
class NotificationManager(ABC):
    def send_notification(self, alert_data: Dict[str, Any]) -> None:
    def log_notification(self, alert_data: Dict[str, Any]) -> None:
```

---

### **3️⃣ reporting/** - گزارش‌دهی
#### 🔹 `report_generator.py`
```python
class ReportGenerator(ABC):
    def generate_report(self, report_data: Dict[str, Any]) -> str:
    def export_report(self, report: str, format_type: str) -> None:
```
#### 🔹 `trend_analyzer.py`
```python
class TrendAnalyzer(ABC):
    def analyze_trends(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
    def predict_future_trends(self, trend_data: Dict[str, Any]) -> Dict[str, Any]:
```
#### 🔹 `dashboard_manager.py`
```python
class DashboardManager(ABC):
    def update_dashboard(self, dashboard_data: Dict[str, Any]) -> None:
    def get_dashboard_snapshot(self) -> Dict[str, Any]:
```

---

✅ **تمامی کلاس‌های ماژول `monitoring/` به صورت کامل پیاده‌سازی و مستند شدند.** 🎉





# 📌 مستندات ماژول `prediction/` در Balance

## 📂 ساختار ماژول
```
balance/
└── prediction/                     # پیش‌بینی و تحلیل آینده
    ├── demand/                     # پیش‌بینی نیازها
    │   ├── model_needs_predictor.py # پیش‌بینی نیاز مدل‌ها
    │   ├── resource_predictor.py    # پیش‌بینی منابع
    │   └── load_predictor.py        # پیش‌بینی بار سیستم
    │
    ├── pattern/                    # تحلیل الگوها
    │   ├── usage_analyzer.py        # تحلیل الگوهای استفاده
    │   ├── behavior_analyzer.py     # تحلیل رفتار سیستم
    │   └── trend_detector.py        # تشخیص روندها
    │
    └── optimization/               # بهینه‌سازی پیش‌بینی
        ├── prediction_tuner.py      # تنظیم پیش‌بینی‌ها
        ├── accuracy_monitor.py      # پایش دقت
        └── model_optimizer.py       # بهینه‌سازی مدل‌های پیش‌بینی
```

---

## 📌 جزئیات کلاس‌ها و متدهای ماژول `prediction/`

### **1️⃣ demand/** - پیش‌بینی نیازهای مدل‌ها و سیستم
#### 🔹 `model_needs_predictor.py`
```python
class ModelNeedsPredictor:
    def predict_needs(self, model_id: str, recent_requests: list) -> dict:
```
📌 **کاربرد**: پیش‌بینی میزان داده‌ای که هر مدل در آینده نیاز خواهد داشت.

#### 🔹 `resource_predictor.py`
```python
class ResourcePredictor:
    def predict_resources(self, model_id: str, recent_usage: list) -> dict:
```
📌 **کاربرد**: پیش‌بینی منابع مورد نیاز (CPU، RAM، Storage) برای مدل‌ها.

#### 🔹 `load_predictor.py`
```python
class LoadPredictor:
    def predict_load(self, time_window: int = 60) -> dict:
```
📌 **کاربرد**: پیش‌بینی میزان بار پردازشی سیستم و جلوگیری از فشار بیش از حد.

---

### **2️⃣ pattern/** - تحلیل الگوهای مصرف و رفتار سیستم
#### 🔹 `usage_analyzer.py`
```python
class UsageAnalyzer:
    def analyze_usage(self, model_id: str, time_window: int = 60) -> dict:
```
📌 **کاربرد**: تحلیل روند مصرف داده‌ها و منابع برای هر مدل.

#### 🔹 `behavior_analyzer.py`
```python
class BehaviorAnalyzer:
    def analyze_behavior(self, model_id: str, time_window: int = 300) -> dict:
```
📌 **کاربرد**: تحلیل رفتار مدل‌ها و شناسایی الگوهای غیرعادی در مصرف منابع.

#### 🔹 `trend_detector.py`
```python
class TrendDetector:
    def detect_trend(self, model_id: str, time_window: int = 86400) -> dict:
```
📌 **کاربرد**: شناسایی روندهای بلندمدت مصرف داده‌ها و منابع در مدل‌ها.

---

### **3️⃣ optimization/** - بهینه‌سازی پیش‌بینی‌ها و عملکرد مدل‌ها
#### 🔹 `prediction_tuner.py`
```python
class PredictionTuner:
    def tune_predictions(self, model_id: str, prediction_data: dict) -> dict:
```
📌 **کاربرد**: تنظیم و بهینه‌سازی پیش‌بینی‌ها برای بهبود عملکرد مدل‌ها.
#### 🔹 `accuracy_monitor.py`
```python
class AccuracyMonitor:
    def monitor_accuracy(self, model_id: str, actual_data: dict, predicted_data: dict) -> dict:
```
📌 **کاربرد**: پایش دقت پیش‌بینی‌ها و تحلیل عملکرد مدل‌های پیش‌بینی.

#### 🔹 `model_optimizer.py`
```python
class ModelOptimizer:
    def optimize_model(self, model_id: str, accuracy_data: dict) -> dict:
```
📌 **کاربرد**: بهینه‌سازی مدل‌های پیش‌بینی برای افزایش دقت و عملکرد آن‌ها.

---

✅ **تمامی کلاس‌های ماژول `prediction/` به صورت کامل پیاده‌سازی و مستند شدند.** 🎉


### ✅ **مستندات ماژول `core/` در Balance**  
📌 این مستند **ماژول `core/`** را با ساختاری مشابه مستند `ai_balance_inside.md` توصیف می‌کند. این ماژول شامل **تحلیل، زمان‌بندی، تخصیص منابع و هماهنگ‌سازی عملیات‌ها** در سیستم Balance است.

---

## 📂 **ساختار ماژول**
```
balance/
└── core/                              # هسته اصلی پردازش‌ها
    ├── analyzer/                      # تحلیل داده‌ها و نیازها
    │   ├── requirement_analyzer.py    # تحلیل نیازهای داده‌ای
    │   ├── distribution_analyzer.py   # تحلیل توزیع داده‌ها
    │   ├── quality_analyzer.py        # تحلیل کیفیت داده‌ها
    │   └── impact_analyzer.py         # تحلیل تأثیر داده‌ها
    │
    ├── scheduler/                     # مدیریت زمان‌بندی وظایف و تخصیص منابع
    │   ├── task_scheduler.py          # زمان‌بندی وظایف
    │   ├── resource_allocator.py      # تخصیص منابع
    │   ├── priority_manager.py        # مدیریت اولویت‌ها
    │   └── dependency_resolver.py     # حل وابستگی‌ها
    │
    ├── coordinator/                   # هماهنگ‌سازی عملیات و پردازش‌ها
    │   ├── operation_coordinator.py   # هماهنگ‌کننده وظایف عملیاتی
    │   ├── model_coordinator.py       # هماهنگ‌کننده مدل‌ها
    │   ├── data_coordinator.py        # هماهنگ‌کننده داده‌ها
    │   └── sync_manager.py            # مدیریت همگام‌سازی داده‌ها
```

---

## 📌 **جزئیات کلاس‌ها و متدهای ماژول `core/`**  

### **1️⃣ analyzer/** - تحلیل داده‌ها و نیازها  

#### 🔹 `requirement_analyzer.py`
```python
class RequirementAnalyzer:
    def log_data_request(self, model_id: str, data_size: int) -> None:
    def analyze_needs(self, model_id: str, recent_requests: List[Dict[str, Any]]) -> Dict[str, Any]:
    def detect_data_shortage(self, model_id: str, data_threshold: int) -> bool:
```
📌 **کاربرد**: ثبت درخواست‌های داده، تحلیل نیازهای داده‌ای مدل‌ها و تشخیص کمبود داده  

#### 🔹 `distribution_analyzer.py`
```python
class DistributionAnalyzer:
    def log_data_usage(self, model_id: str, data_size: int) -> None:
    def analyze_distribution(self) -> Dict[str, Any]:
    def detect_imbalance(self, threshold: float = 1.5) -> List[str]:
```
📌 **کاربرد**: بررسی توزیع داده بین مدل‌ها و شناسایی عدم تعادل در مصرف داده‌ها  

#### 🔹 `quality_analyzer.py`
```python
class QualityAnalyzer:
    def log_data_quality(self, model_id: str, total_records: int, invalid_records: int, noise_level: float) -> None:
    def analyze_quality(self) -> Dict[str, Any]:
    def detect_low_quality_models(self, invalid_threshold: float = 0.1, noise_threshold: float = 0.5) -> List[str]:
```
📌 **کاربرد**: تحلیل کیفیت داده‌های پردازش‌شده و شناسایی مدل‌هایی که داده‌های نامعتبر دریافت کرده‌اند  

#### 🔹 `impact_analyzer.py`
```python
class ImpactAnalyzer:
    def log_impact(self, model_id: str, performance_change: float, resource_usage: int, quality_shift: float) -> None:
    def analyze_impact(self) -> Dict[str, Any]:
    def detect_high_impact_models(self, performance_threshold: float = 5.0, resource_threshold: int = 1000, quality_threshold: float = 0.2) -> List[str]:
```
📌 **کاربرد**: بررسی تأثیر تغییرات داده‌ها بر عملکرد مدل‌ها و شناسایی مدل‌هایی که بیشترین تأثیر را پذیرفته‌اند  

---

### **2️⃣ scheduler/** - زمان‌بندی وظایف و تخصیص منابع  

#### 🔹 `task_scheduler.py`
```python
class TaskScheduler:
    def schedule_task(self, task: Callable, priority: int = 1, delay: float = 0, *args, **kwargs) -> None:
    def execute_next_task(self) -> bool:
    def run_scheduler(self) -> None:
    def get_scheduled_tasks(self) -> List[Dict[str, Any]]:
```
📌 **کاربرد**: مدیریت زمان‌بندی وظایف و اجرای آن‌ها بر اساس اولویت و منابع  

#### 🔹 `resource_allocator.py`
```python
class ResourceAllocator:
    def get_system_resources(self) -> Dict[str, Any]:
    def can_allocate_resources(self, cpu_demand: float, memory_demand: float, gpu_demand: float = 0.0) -> bool:
    def allocate_resources(self, cpu_demand: float, memory_demand: float, gpu_demand: float = 0.0) -> Dict[str, Any]:
```
📌 **کاربرد**: تخصیص منابع پردازشی به وظایف با بررسی وضعیت منابع سیستم  

#### 🔹 `priority_manager.py`
```python
class PriorityManager:
    def set_priority(self, task_id: str, priority: int) -> None:
    def get_priority(self, task_id: str) -> int:
    def adjust_priority(self, task_id: str, adjustment: int) -> None:
    def get_sorted_tasks(self) -> List[Tuple[str, int]]:
```
📌 **کاربرد**: تنظیم و مدیریت اولویت وظایف پردازشی  

#### 🔹 `dependency_resolver.py`
```python
class DependencyResolver:
    def add_dependency(self, task_id: str, dependencies: List[str]) -> None:
    def get_dependencies(self, task_id: str) -> List[str]:
    def can_execute(self, task_id: str) -> bool:
    def resolve_dependencies(self, task_id: str) -> List[str]:
```
📌 **کاربرد**: بررسی و مدیریت وابستگی‌های وظایف پردازشی  

---

### **3️⃣ coordinator/** - هماهنگ‌سازی عملیات و پردازش‌ها  

#### 🔹 `operation_coordinator.py`
```python
class OperationCoordinator:
    def execute_task(self, task_id: str, task_func: callable, cpu_demand: float, memory_demand: float, priority: int = 1) -> Dict[str, Any]:
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
```
📌 **کاربرد**: هماهنگ‌سازی اجرای وظایف و تخصیص منابع  

#### 🔹 `model_coordinator.py`
```python
class ModelCoordinator:
    def register_model(self, model_id: str, cpu_demand: float, memory_demand: float, priority: int = 1) -> None:
    def allocate_resources_for_model(self, model_id: str) -> Dict[str, Any]:
    def get_model_status(self, model_id: str) -> Dict[str, Any]:
    def release_resources(self, model_id: str) -> None:
```
📌 **کاربرد**: هماهنگی و تخصیص منابع بین مدل‌های پردازشی  

#### 🔹 `data_coordinator.py`
```python
class DataCoordinator:
    def register_data_source(self, data_id: str, size: int, source: str) -> None:
    def allocate_data(self, data_id: str) -> Dict[str, Any]:
    def release_data(self, data_id: str) -> None:
    def synchronize_data(self, data_id: str, destination: str) -> Dict[str, Any]:
```
📌 **کاربرد**: مدیریت تخصیص و همگام‌سازی داده‌ها بین منابع  

#### 🔹 `sync_manager.py`
```python
class SyncManager:
    def register_sync_task(self, sync_id: str, source: str, destination: str, size: int) -> None:
    def execute_sync(self, sync_id: str) -> Dict[str, Any]:
    def get_sync_status(self, sync_id: str) -> Dict[str, Any]:
```
📌 **کاربرد**: مدیریت همگام‌سازی اطلاعات و پردازش‌های سیستمی  

---

✅ **تمامی کلاس‌های ماژول `core/` به صورت کامل پیاده‌سازی و مستند شدند.** 🎉




# 📌 مستندات ماژول `batch/` در Balance

## 📂 **ساختار ماژول**
```
balance/
└── batch/                          # مدیریت پردازش دسته‌ای
    ├── processor/                  # پردازش دسته‌ها
    │   ├── batch_processor.py      # پردازشگر دسته‌ای
    │   ├── request_merger.py       # ادغام درخواست‌ها
    │   └── batch_splitter.py       # تقسیم دسته‌ها
    │
    ├── optimizer/                  # بهینه‌سازی دسته‌ها
    │   ├── batch_optimizer.py      # بهینه‌سازی ترکیب دسته‌ها
    │   ├── size_calculator.py      # محاسبه اندازه بهینه دسته‌ها
    │   └── efficiency_analyzer.py  # تحلیل و ارزیابی کارایی
    │
    └── scheduler/                  # زمان‌بندی دسته‌ها
        ├── batch_scheduler.py      # زمان‌بند دسته‌ای
        ├── priority_handler.py     # مدیریت اولویت دسته‌ها
        └── resource_manager.py     # مدیریت منابع دسته‌ها
```

---

## 📌 **جزئیات کلاس‌ها و متدهای ماژول `batch/`**

### **1️⃣ processor/** - پردازش دسته‌ای
#### 🔹 `batch_processor.py`
```python
class BatchProcessor:
    def process_batch(self, batch_data: List[Dict[str, Any]], priority: int = 1) -> Dict[str, Any]:
    def handle_incoming_batches(self, batch_queue: asyncio.Queue):
```
📌 **کاربرد**: مدیریت پردازش دسته‌ای داده‌ها، ترکیب درخواست‌های مشابه و تخصیص منابع.

#### 🔹 `request_merger.py`
```python
class RequestMerger:
    def merge_requests(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
```
📌 **کاربرد**: ادغام درخواست‌های مشابه برای کاهش پردازش‌های تکراری.

#### 🔹 `batch_splitter.py`
```python
class BatchSplitter:
    def split_large_batches(self, batch_data: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
```
📌 **کاربرد**: تقسیم دسته‌های بزرگ به دسته‌های کوچک‌تر برای پردازش بهینه‌تر.

---

### **2️⃣ optimizer/** - بهینه‌سازی پردازش دسته‌ای
#### 🔹 `batch_optimizer.py`
```python
class BatchOptimizer:
    def calculate_optimal_resources(self, batch_data: List[Dict[str, Any]]) -> Dict[str, float]:
    def optimize_batch_composition(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
```
📌 **کاربرد**: بهینه‌سازی ترکیب دسته‌ها و تخصیص منابع پردازشی.

#### 🔹 `size_calculator.py`
```python
class SizeCalculator:
    def calculate_batch_size(self, previous_batches: List[Dict[str, Any]]) -> int:
```
📌 **کاربرد**: محاسبه اندازه بهینه دسته بر اساس داده‌های پردازش قبلی.

#### 🔹 `efficiency_analyzer.py`
```python
class EfficiencyAnalyzer:
    def analyze_efficiency(self, batch_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
```
📌 **کاربرد**: تحلیل کارایی دسته‌های پردازشی و شناسایی دسته‌های ناکارآمد.

---

### **3️⃣ scheduler/** - زمان‌بندی پردازش دسته‌ای
#### 🔹 `batch_scheduler.py`
```python
class BatchScheduler:
    async def schedule_batch(self, batch_data: Dict[str, Any], priority: int = 1):
    async def process_batches(self, process_function):
```
📌 **کاربرد**: زمان‌بندی دسته‌ها و مدیریت صف پردازش بر اساس اولویت.

#### 🔹 `priority_handler.py`
```python
class PriorityHandler:
    def assign_priority(self, batch_data: Dict[str, Any]) -> int:
    def adjust_priority(self, batch_id: str, new_priority: str) -> int:
```
📌 **کاربرد**: تنظیم و مدیریت اولویت دسته‌های پردازشی.

#### 🔹 `resource_manager.py`
```python
class ResourceManager:
    def allocate_resources(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
    def release_resources(self, used_resources: Dict[str, float]) -> None:
```
📌 **کاربرد**: مدیریت تخصیص و آزادسازی منابع پردازشی برای دسته‌ها.

---

✅ **ماژول `batch/` به‌صورت کامل مستندسازی و پیاده‌سازی شد.** 🎉



# 📌 مستندات ماژول `quality/` در Balance

## 📂 **ساختار ماژول**
```
quality/
├── monitor/                    # پایش کیفیت داده‌ها
│   ├── quality_monitor.py      # پایشگر کیفیت داده‌ها
│   ├── metric_collector.py     # جمع‌آوری متریک‌های کیفیت داده‌ها
│   └── alert_manager.py        # مدیریت هشدارهای کیفیت
│
├── control/                    # کنترل کیفیت داده‌ها
│   ├── quality_controller.py   # کنترل‌کننده کیفیت داده‌ها
│   ├── threshold_manager.py    # مدیریت آستانه‌های کیفیت داده‌ها
│   └── action_executor.py      # اجرای اقدامات اصلاحی بر روی داده‌ها
│
└── improvement/                # بهبود کیفیت داده‌ها
    ├── quality_optimizer.py    # بهینه‌ساز کیفیت داده‌ها
    ├── strategy_selector.py    # انتخاب استراتژی بهبود کیفیت داده‌ها
    └── feedback_analyzer.py    # تحلیل بازخوردهای کیفیت داده‌ها
```

---

## 📌 **جزئیات کلاس‌ها و متدهای ماژول `quality/`**

### **1️⃣ monitor/** - پایش کیفیت داده‌ها
#### 🔹 `quality_monitor.py`
```python
class QualityMonitor:
    def evaluate_data_quality(self, data_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def monitor_trends(self, historical_quality: List[float]) -> Dict[str, Any]:
```
📌 **کاربرد**: بررسی کیفیت داده‌ها و تشخیص کاهش کیفیت در طول زمان.

#### 🔹 `metric_collector.py`
```python
class MetricCollector:
    def collect_metric(self, data_id: str, quality_score: float, processing_time: float) -> None:
    def get_average_metrics(self) -> Dict[str, float]:
    def reset_metrics(self) -> None:
```
📌 **کاربرد**: جمع‌آوری متریک‌های مربوط به کیفیت داده‌ها و پردازش‌ها.

#### 🔹 `alert_manager.py`
```python
class AlertManager:
    def check_for_alerts(self, quality_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def get_alerts(self) -> List[Dict[str, Any]]:
    def clear_alerts(self) -> None:
```
📌 **کاربرد**: مدیریت هشدارها در صورت کاهش کیفیت داده‌ها.

---

### **2️⃣ control/** - کنترل کیفیت داده‌ها
#### 🔹 `quality_controller.py`
```python
class QualityController:
    def validate_data_quality(self, data_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def apply_corrections(self, invalid_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
```
📌 **کاربرد**: بررسی کیفیت داده‌ها و اعمال اقدامات اصلاحی.

#### 🔹 `threshold_manager.py`
```python
class ThresholdManager:
    def is_valid(self, data: Dict[str, Any]) -> bool:
    def adjust_threshold(self, new_threshold: float) -> None:
```
📌 **کاربرد**: مدیریت حداقل استانداردهای کیفیت داده‌ها.

#### 🔹 `action_executor.py`
```python
class ActionExecutor:
    def correct_data(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
```
📌 **کاربرد**: اجرای اقدامات اصلاحی برای بهبود کیفیت داده‌ها.

---

### **3️⃣ improvement/** - بهبود کیفیت داده‌ها
#### 🔹 `quality_optimizer.py`
```python
class QualityOptimizer:
    def optimize_data(self, data_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
```
📌 **کاربرد**: بهینه‌سازی داده‌ها برای افزایش کیفیت پردازش‌ها.

#### 🔹 `strategy_selector.py`
```python
class StrategySelector:
    def select_strategy(self, data: Dict[str, Any]) -> str:
    def apply_strategy(self, data: Dict[str, Any]) -> Dict[str, Any]:
```
📌 **کاربرد**: انتخاب استراتژی مناسب برای بهبود کیفیت داده‌ها.

#### 🔹 `feedback_analyzer.py`
```python
class FeedbackAnalyzer:
    def collect_feedback(self, feedback: Dict[str, Any]) -> None:
    def analyze_feedback(self) -> Dict[str, float]:
    def reset_feedback(self) -> None:
```
📌 **کاربرد**: تحلیل بازخوردهای کیفیت داده‌ها و ارائه پیشنهادات بهبود.

---



# مستندات سرویس‌های ماژول Balance در Smart Whale AI

## 📌 مقدمه
ماژول Balance بخشی از سامانه Smart Whale AI است که مسئول مدیریت و هماهنگی ارتباطات پیام‌رسانی میان اجزای مختلف سیستم می‌باشد. در این ماژول، سرویس‌هایی پیاده‌سازی شده‌اند که وظیفه ارسال درخواست‌های جمع‌آوری داده، مدیریت ارتباط با مدل‌ها، و راه‌اندازی یکپارچه پیام‌رسانی را بر عهده دارند. سرویس‌های این ماژول از قابلیت‌های ماژول Messaging بهره می‌برند تا ارتباطات به صورت امن، استاندارد و بهینه برقرار شود.

---

## 🏗️ ساختار ماژول

دایرکتوری ماژول Balance در بخش services به صورت زیر سازماندهی شده است:

```
ai/balance/services/
├── data_service.py          # سرویس ارسال درخواست‌های جمع‌آوری داده به ماژول Data
├── messaging_service.py     # سرویس مدیریت یکپارچه پیام‌رسانی در ماژول Balance
├── model_service.py         # سرویس مدیریت ارتباط با مدل‌ها و پردازش درخواست‌های آن‌ها
└── __init__.py              # صادرسازی سرویس‌های Balance برای ارتباط با سایر ماژول‌ها
```

---

## 📦 شرح تفصیلی سرویس‌ها

### 1️⃣ سرویس داده (DataService) - فایل `data_service.py`
این سرویس وظیفه ارسال درخواست‌های جمع‌آوری داده از ماژول Balance به ماژول Data را بر عهده دارد. از ویژگی‌های این سرویس می‌توان به موارد زیر اشاره کرد:

- **هدف و وظیفه سرویس:**  
  - فرمت‌بندی و ایجاد درخواست‌های داده با استفاده از توابع و ساختارهای استاندارد پیام (مانند DataRequest).
  - مدیریت اتصال به Kafka و اطمینان از وجود موضوع (Topic) درخواست‌ها از طریق استفاده از TopicManager.
  - پشتیبانی از ارسال درخواست‌های تک و دسته‌ای (batch) به صورت غیرهمزمان.
  
- **ساختار کلاس DataService:**
  - **متغیرهای عضو:**
    - `self.kafka_service`: شیء singleton سرویس Kafka جهت ارسال پیام.
    - `self.topic_manager`: نمونه‌ای از TopicManager برای مدیریت موضوعات Kafka.
    - `self.request_topic`: موضوع اصلی برای ارسال درخواست‌های داده؛ به صورت ثابت با استفاده از DATA_REQUESTS_TOPIC.
    - `self.request_counter`: شمارنده درخواست‌ها جهت پیگیری تعداد درخواست‌های ارسال‌شده.
    - `self._is_initialized`: فلگ داخلی برای تعیین اینکه سرویس اولیه‌سازی شده است یا خیر.
  
  - **متدهای اصلی:**
    - **`initialize()`**  
      آماده‌سازی اولیه سرویس؛ شامل اتصال به Kafka و اطمینان از وجود موضوع درخواست‌ها.  
      _خروجی:_ ثبت پیغام موفقیت در لاگ و تغییر وضعیت _is_initialized_.
    
    - **`request_data(model_id, query, data_type, source_type, priority, request_source, **params)`**  
      ارسال یک درخواست جمع‌آوری داده به ماژول Data.  
      - پارامترها:
        - `model_id`: شناسه مدل درخواست‌کننده.
        - `query`: عبارت جستجو (می‌تواند URL یا عنوان مقاله باشد).
        - `data_type`: نوع داده مورد نیاز (مثلاً TEXT، IMAGE و ...).
        - `source_type`: نوع منبع داده (مثلاً WEB، WIKI، TWITTER و ...).
        - `priority`: سطح اولویت درخواست.
        - `request_source`: منبع درخواست (USER، MODEL یا SYSTEM).
        - `**params`: سایر پارامترهای اختیاری منطبق با نیاز هر منبع.
      - اقدامات:
        - آماده‌سازی سرویس (در صورت عدم اولیه‌سازی).
        - تبدیل مقادیر رشته‌ای به Enum‌های مربوطه با اعتبارسنجی.
        - ایجاد موضوع نتایج اختصاصی برای مدل از طریق TopicManager.
        - تنظیم پارامترهای خاص هر منبع؛ مانند تنظیم زبان پیش‌فرض برای ویکی‌پدیا یا تشخیص URL برای منابع وب.
        - افزایش شمارنده درخواست.
        - ایجاد یک DataRequest با استفاده از تابع کمکی create_data_request.
        - ارسال درخواست از طریق kafka_service.
      - _خروجی:_ دیکشنری شامل وضعیت ارسال، شناسه درخواست، زمان ارسال، اطلاعات مدل، نوع داده، منبع و موضوع پاسخ؛ همچنین زمان تخمینی پردازش که توسط متد `_estimate_processing_time()` برآورد می‌شود.
      
    - **`request_batch(model_id, queries)`**  
      ارسال چندین درخواست به صورت دسته‌ای.  
      - دریافت لیست دیکشنری‌های حاوی اطلاعات هر درخواست.
      - فراخوانی متد `request_data` برای هر درخواست.
      - _خروجی:_ لیست نتایج ارسال درخواست‌های دسته‌ای.
      
    - **`_estimate_processing_time(source_type, parameters)`**  
      تخمین زمان پردازش درخواست بر اساس نوع منبع و پارامترهای ارسال شده.  
      - بر اساس نوع منبع مانند WIKI، WEB، شبکه‌های اجتماعی (TWITTER، TELEGRAM) یا منابع ویدیویی (YOUTUBE، APARAT) زمان‌های متفاوتی برآورد می‌شود.
      - _خروجی:_ زمان تخمینی به ثانیه.

---

### 2️⃣ سرویس پیام‌رسانی (MessagingService) - فایل `messaging_service.py`
این سرویس مسئول راه‌اندازی، نظارت و مدیریت یکپارچه ارتباطات پیام‌رسانی در ماژول Balance است. از ویژگی‌های این سرویس می‌توان به موارد زیر اشاره کرد:

- **هدف و وظیفه سرویس:**
  - هماهنگی بین سرویس‌های داده و مدل به منظور ایجاد یک سیستم پیام‌رسانی یکپارچه.
  - راه‌اندازی اولیه اتصال به Kafka و اطمینان از وجود موضوعات اصلی (مانند موضوعات درخواست‌های مدل، رویدادها و متریک‌ها).
  - فراهم آوردن متدهایی جهت ثبت مدل، ارسال درخواست داده، انتشار رویدادها و متریک‌ها.
  - ایجاد یک حلقه اصلی اجرای سرویس که در انتظار سیگنال توقف قرار دارد.

- **ساختار کلاس MessagingService:**
  - **متغیرهای عضو:**
    - `self.kafka_service`: شیء KafkaService جهت ارسال و دریافت پیام.
    - `self.topic_manager`: نمونه TopicManager جهت مدیریت موضوعات.
    - `self.data_service`: ارجاع به سرویس داده جهت ارسال درخواست‌های داده.
    - `self.model_service`: ارجاع به سرویس مدل جهت مدیریت ارتباط با مدل‌ها.
    - `self._is_initialized`: فلگ تعیین اولیه‌سازی سرویس.
    - `self._shutdown_event`: شیء asyncio.Event جهت توقف حلقه اصلی سرویس.
  
  - **متدهای اصلی:**
    - **`initialize()`**  
      آماده‌سازی اولیه سرویس پیام‌رسانی؛ شامل اتصال به Kafka، اطمینان از ایجاد موضوعات اصلی، راه‌اندازی سرویس‌های وابسته (data_service و model_service) و اشتراک در موضوع درخواست‌های مدل‌ها.
    
    - **`register_model(model_id, handler)`**  
      ثبت یک مدل جدید در سیستم و اشتراک در موضوع نتایج اختصاصی آن.  
      - ثبت مدل در سرویس مدل.
      - اشتراک در موضوع نتایج مدل با استفاده از handler اختیاری جهت پردازش نتایج.
      - _خروجی:_ دیکشنری شامل شناسه مدل، وضعیت ثبت و موضوع مربوطه.
    
    - **`request_data(model_id, query, data_type, source_type, priority, request_source, **params)`**  
      متدی مشابه متد موجود در DataService؛ وظیفه ارسال درخواست جمع‌آوری داده از یک منبع مشخص را بر عهده دارد.  
      - قبل از ارسال درخواست، اطمینان حاصل می‌شود که مدل ثبت شده است.
      - درخواست از طریق data_service ارسال می‌شود.
      - _خروجی:_ نتیجه درخواست ارسال شده به صورت دیکشنری.
    
    - **`publish_event(event_type, event_data)`**  
      انتشار یک رویداد در سیستم؛ با افزودن زمان فعلی به داده‌های رویداد و ارسال آن به موضوع BALANCE_EVENTS_TOPIC.
      - _خروجی:_ نتیجه ارسال رویداد (True/False).
    
    - **`publish_metric(metric_name, metric_data)`**  
      انتشار یک متریک؛ با افزودن نام متریک و زمان فعلی به داده‌های متریک و ارسال به موضوع BALANCE_METRICS_TOPIC.
      - _خروجی:_ نتیجه ارسال متریک.
    
    - **`run()`**  
      شروع اجرای سرویس پیام‌رسانی؛ ورود به حلقه اصلی که تا دریافت سیگنال توقف اجرا می‌شود.
    
    - **`shutdown()`**  
      توقف سرویس پیام‌رسانی و آزادسازی منابع؛ شامل قطع اتصال از Kafka.
    
    - **`_get_timestamp()`**  
      تولید رشته زمان فعلی به صورت ISO برای ثبت در داده‌های پیام.

---

### 3️⃣ سرویس مدل (ModelService) - فایل `model_service.py`
این سرویس مسئول مدیریت ارتباط با مدل‌ها، پردازش درخواست‌های دریافتی از مدل‌ها و مدیریت اشتراک‌های اختصاصی مدل‌ها در موضوعات Kafka می‌باشد.

- **هدف و وظیفه سرویس:**
  - ثبت مدل‌ها در سیستم و ایجاد موضوع نتایج اختصاصی برای هر مدل.
  - دریافت و پردازش درخواست‌های مدل‌ها (مانند درخواست جمع‌آوری داده) و ارسال آنها به سرویس داده.
  - اشتراک در موضوع درخواست‌های مدل‌ها و دریافت پاسخ‌ها از مدل‌ها.
  - فراهم آوردن متدهایی جهت ارسال مستقیم نتایج به مدل‌ها (forwarding).

- **ساختار کلاس ModelService:**
  - **متغیرهای عضو:**
    - `self.kafka_service`: شیء KafkaService جهت ارسال و دریافت پیام.
    - `self.topic_manager`: نمونه TopicManager جهت مدیریت موضوعات.
    - `self.models_requests_topic`: موضوع اصلی برای درخواست‌های مدل‌ها؛ از طریق MODELS_REQUESTS_TOPIC تعریف شده است.
    - `self.registered_models`: مجموعه‌ای از مدل‌های ثبت‌شده در سیستم.
    - `self.model_handlers`: دیکشنری نگهدارنده پردازشگر اختصاصی (handler) برای هر مدل.
    - `self._is_initialized`: فلگ اولیه‌سازی سرویس.
  
  - **متدهای اصلی:**
    - **`initialize()`**  
      آماده‌سازی اولیه سرویس مدل؛ شامل اتصال به Kafka، اطمینان از وجود موضوع درخواست‌های مدل‌ها و آماده‌سازی سرویس داده.
    
    - **`register_model(model_id, handler)`**  
      ثبت یک مدل جدید؛ ایجاد موضوع نتایج اختصاصی برای مدل و ذخیره پردازشگر اختصاصی (در صورت ارائه handler).  
      - افزودن مدل به مجموعه registered_models.
      - _خروجی:_ دیکشنری شامل شناسه مدل، وضعیت ثبت و موضوع اختصاصی.
    
    - **`unregister_model(model_id)`**  
      حذف ثبت یک مدل از سیستم؛ شامل حذف از مجموعه registered_models و پاکسازی پردازشگر اختصاصی.
    
    - **`process_model_request(request_data)`**  
      پردازش درخواست‌های دریافتی از مدل‌ها؛ شامل:
      - اعتبارسنجی درخواست.
      - ساخت نمونه DataRequest از داده‌های دریافتی.
      - استخراج اطلاعات مانند model_id، نوع عملیات، نوع داده و پارامترها.
      - در صورت عدم ثبت مدل، ثبت خودکار مدل.
      - در صورت درخواست جمع‌آوری داده (FETCH_DATA)، ارسال درخواست به سرویس داده.
      - در سایر موارد، ثبت هشدار برای عملیات‌های پشتیبانی نشده.
    
    - **`subscribe_to_model_requests()`**  
      اشتراک در موضوع درخواست‌های مدل‌ها جهت دریافت و پردازش درخواست‌های ورودی.  
      - تعریف یک تابع پردازشگر (request_handler) جهت پردازش پیام‌های دریافتی.
      - استفاده از kafka_service جهت اشتراک در موضوع MODELS_REQUESTS_TOPIC با group_id مشخص.
    
    - **`subscribe_to_model_results(model_id, handler)`**  
      اشتراک در موضوع نتایج اختصاصی یک مدل جهت دریافت پاسخ‌های جمع‌آوری داده.  
      - ثبت مدل در صورت عدم وجود.
      - به‌روزرسانی پردازشگر اختصاصی (handler) در صورت ارائه.
      - اشتراک در موضوع نتایج مدل با group_id منحصر به فرد.
    
    - **`forward_result_to_model(model_id, result_data)`**  
      ارسال مستقیم نتیجه به یک مدل از طریق موضوع اختصاصی آن.  
      - دریافت موضوع اختصاصی مدل از TopicManager.
      - ارسال پیام به آن موضوع از طریق kafka_service.
      - _خروجی:_ نتیجه ارسال (True/False).

---

### 4️⃣ صادرسازی سرویس‌ها (__init__.py) - فایل `__init__.py`
این فایل تمامی سرویس‌های مربوط به ماژول Balance را صادرسازی می‌کند تا سایر بخش‌های سیستم بتوانند به آسانی به آن‌ها دسترسی داشته باشند.

- **صادرسازی سرویس‌ها:**
  - `DataService` و نمونه singleton آن `data_service`
  - `ModelService` و نمونه singleton آن `model_service`
  - `MessagingService` و نمونه singleton آن `messaging_service`
  
- **لیست نمادهای صادرشده:**  
  آرایه __all__ شامل نام‌های سرویس‌های صادر شده جهت استفاده در سایر ماژول‌ها.

---

## 🔄 تعامل بین اجزا در ماژول Balance

1. **تعامل سرویس‌های داده، مدل و پیام‌رسانی:**
   - **DataService** درخواست‌های جمع‌آوری داده را فرمت‌بندی و از طریق Kafka ارسال می‌کند.
   - **ModelService** وظیفه دریافت درخواست‌های مدل، پردازش آنها و ارسال نتایج به مدل‌ها را بر عهده دارد.
   - **MessagingService** به عنوان سرویس مرکزی، هماهنگی بین سرویس‌های داده و مدل را انجام داده و امکان انتشار رویدادها و متریک‌ها را فراهم می‌آورد.

2. **استفاده از TopicManager و KafkaService:**
   - TopicManager مسئول اطمینان از ایجاد و مدیریت موضوعات مورد نیاز (برای درخواست‌ها، نتایج، رویدادها و متریک‌ها) می‌باشد.
   - KafkaService به عنوان لایه ارتباطی، پیام‌ها را به صورت غیرهمزمان ارسال و دریافت می‌کند.

3. **اعتبارسنجی و ثبت مدل‌ها:**
   - قبل از ارسال هر درخواست، ثبت مدل در سیستم از طریق ModelService انجام می‌شود تا اطمینان حاصل شود که مدل دارای موضوع اختصاصی برای دریافت نتایج است.
   - توابعی جهت ثبت، لغو ثبت و مدیریت پردازشگرهای اختصاصی برای هر مدل فراهم شده است.

---

## 📊 استراتژی‌های بهینه‌سازی و عملکردی

### 1. بهینه‌سازی عملکرد درخواست‌ها
- **اتصال غیرهمزمان به Kafka:**  
  استفاده از متدهای async/await در تمامی سرویس‌ها باعث می‌شود درخواست‌ها به صورت غیرهمزمان ارسال و دریافت شوند.
- **ارسال دسته‌ای (Batch) درخواست‌ها:**  
  متد `request_batch` در DataService امکان ارسال چندین درخواست به صورت دسته‌ای را فراهم می‌کند.
- **تخمین زمان پردازش:**  
  متد `_estimate_processing_time` زمان تقریبی پردازش درخواست‌ها را بر اساس نوع منبع و پارامترهای ارسال شده برآورد می‌کند.

### 2. بهبود هماهنگی میان سرویس‌ها
- **ثبت و اشتراک‌گذاری مدل‌ها:**  
  سرویس ModelService از طریق متدهای `register_model` و `subscribe_to_model_results` اطمینان حاصل می‌کند که هر مدل دارای کانال ارتباطی اختصاصی است.
- **هماهنگی سرویس پیام‌رسانی:**  
  MessagingService به عنوان نقطه مرکزی هماهنگی، همه سرویس‌های وابسته را راه‌اندازی و مدیریت می‌کند و امکان انتشار رویدادها و متریک‌ها را بهبود می‌بخشد.

### 3. مدیریت خطا و پایداری
- **اعتبارسنجی پیام‌ها:**  
  استفاده از توابع اعتبارسنجی در DataService و ModelService از بروز خطاهای ساختاری جلوگیری می‌کند.
- **مدیریت اشتراک‌ها:**  
  اشتراک‌گذاری و لغو اشتراک‌ها از طریق KafkaService تضمین می‌کند که پیام‌ها به درستی پردازش شده و منابع به موقع آزاد می‌شوند.
- **ثبت لاگ‌های دقیق:**  
  استفاده از logging در تمامی متدها باعث نظارت دقیق بر عملکرد سرویس‌ها و رفع سریع مشکلات احتمالی می‌شود.

---



