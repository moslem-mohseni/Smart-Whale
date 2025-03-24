# 📌 مستندات ماژول `routing/` در Federation

## 📂 ساختار ماژول
```
federation/
├── routing/                           # مسیریابی هوشمند درخواست‌ها
│   ├── dispatcher/                    # توزیع و مدیریت درخواست‌ها
│   │   ├── request_dispatcher.py      # توزیع‌کننده درخواست‌ها بین مدل‌ها
│   │   ├── load_balancer.py           # متعادل‌کننده بار بین مدل‌ها
│   │   └── priority_handler.py        # مدیریت اولویت درخواست‌ها
│   │
│   ├── optimizer/                     # بهینه‌سازی مسیرهای ارتباطی
│   │   ├── route_optimizer.py         # انتخاب مسیر بهینه برای پردازش
│   │   ├── path_analyzer.py           # تحلیل مسیرهای گذشته
│   │   └── cost_calculator.py         # محاسبه هزینه مسیرها
│   │
│   └── predictor/                      # پیش‌بینی بار پردازشی و مدیریت پیش‌بارگذاری
│       ├── demand_predictor.py        # پیش‌بینی میزان درخواست‌های آینده
│       ├── pattern_analyzer.py        # تحلیل الگوهای استفاده
│       └── preload_manager.py         # مدیریت پیش‌بارگذاری مدل‌ها
```

---

## 📌 جزئیات کلاس‌ها و متدهای ماژول `routing/`

### **1️⃣ dispatcher/** - توزیع درخواست‌ها بین مدل‌ها
#### 🔹 `request_dispatcher.py`
```python
class RequestDispatcher:
    def dispatch_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
    def _send_to_model(self, model: str, request: Dict[str, Any]) -> Dict[str, Any]:
```
📌 **کاربرد**: مدیریت و توزیع درخواست‌ها بین مدل‌های موجود و ارسال آنها به مدل مناسب.

#### 🔹 `load_balancer.py`
```python
class LoadBalancer:
    def select_model(self, request: Dict[str, Any]) -> Optional[str]:
    def release_model(self, model: str) -> None:
```
📌 **کاربرد**: متعادل‌سازی بار بین مدل‌ها و انتخاب بهینه مدل برای اجرای درخواست.

#### 🔹 `priority_handler.py`
```python
class PriorityHandler:
    def evaluate_priority(self, request: Dict[str, Any]) -> int:
    def compare_priority(self, req1: Dict[str, Any], req2: Dict[str, Any]) -> int:
```
📌 **کاربرد**: تعیین اولویت درخواست‌ها و تخصیص منابع بر اساس اولویت آنها.

---

### **2️⃣ optimizer/** - بهینه‌سازی مسیرهای ارتباطی بین مدل‌ها
#### 🔹 `route_optimizer.py`
```python
class RouteOptimizer:
    def optimize_route(self, request: Dict[str, Any], available_routes: List[str]) -> str:
```
📌 **کاربرد**: تحلیل مسیرها و انتخاب مسیر بهینه برای مسیریابی درخواست‌ها.

#### 🔹 `path_analyzer.py`
```python
class PathAnalyzer:
    def record_path(self, path: str) -> None:
    def get_most_frequent_paths(self, top_n: int = 3) -> List[str]:
```
📌 **کاربرد**: تحلیل مسیرهای پردازشی گذشته و شناسایی مسیرهای پرتکرار برای بهینه‌سازی.

#### 🔹 `cost_calculator.py`
```python
class CostCalculator:
    def calculate_cost(self, route: str, request: Dict[str, Any]) -> float:
    def get_cheapest_route(self, routes: Dict[str, Dict[str, Any]]) -> str:
```
📌 **کاربرد**: محاسبه هزینه مسیرهای پردازشی و انتخاب مسیر کم‌هزینه‌تر.

---

### **3️⃣ predictor/** - پیش‌بینی بار پردازشی و مدیریت پیش‌بارگذاری مدل‌ها
#### 🔹 `demand_predictor.py`
```python
class DemandPredictor:
    def record_request(self, request_type: str) -> None:
    def predict_demand(self) -> Dict[str, int]:
```
📌 **کاربرد**: پیش‌بینی میزان درخواست‌های آینده و نیاز پردازشی مدل‌ها.

#### 🔹 `pattern_analyzer.py`
```python
class PatternAnalyzer:
    def record_request(self, request_type: str) -> None:
    def analyze_patterns(self) -> Dict[str, float]:
```
📌 **کاربرد**: تحلیل الگوهای استفاده از مدل‌ها برای پیش‌بینی روندهای آینده.

#### 🔹 `preload_manager.py`
```python
class PreloadManager:
    def preload_model(self, model_name: str) -> None:
    def get_preloaded_models(self) -> List[str]:
    def is_model_preloaded(self, model_name: str) -> bool:
```
📌 **کاربرد**: مدیریت پیش‌بارگذاری مدل‌ها برای کاهش تأخیر و افزایش سرعت پردازش درخواست‌ها.

---

✅ **تمامی کلاس‌های ماژول `routing/` به صورت کامل پیاده‌سازی و مستند شدند.** 🎉





# 📌 مستندات ماژول `knowledge_sharing/` در Federation

## 📂 ساختار ماژول
```
federation/
├── knowledge_sharing/                 # اشتراک دانش بین مدل‌ها
│   ├── manager/                       # مدیریت کلی اشتراک دانش
│   │   ├── knowledge_manager.py       # مدیریت ذخیره و بازیابی دانش
│   │   ├── sharing_optimizer.py       # بهینه‌سازی فرآیند اشتراک دانش
│   │   └── privacy_guard.py           # حفاظت از حریم خصوصی داده‌های اشتراکی
│   │
│   ├── sync/                          # هماهنگ‌سازی و کنترل نسخه دانش
│   │   ├── sync_manager.py            # مدیریت همگام‌سازی دانش بین مدل‌ها
│   │   ├── conflict_resolver.py       # حل تعارض‌های داده‌ای
│   │   └── version_controller.py      # مدیریت نسخه‌های مختلف دانش
│   │
│   └── transfer/                      # انتقال داده‌های فشرده و بررسی یکپارچگی
│       ├── data_compressor.py         # فشرده‌سازی داده‌های اشتراکی
│       ├── efficient_transfer.py      # انتقال بهینه دانش بین مدل‌ها
│       └── integrity_checker.py       # بررسی صحت و یکپارچگی داده‌های منتقل‌شده
```

---

## 📌 جزئیات کلاس‌ها و متدهای ماژول `knowledge_sharing/`

### **1️⃣ manager/** - مدیریت کلی اشتراک دانش
#### 🔹 `knowledge_manager.py`
```python
class KnowledgeManager:
    def store_knowledge(self, model_name: str, knowledge: Any) -> None:
    def retrieve_knowledge(self, model_name: str) -> Any:
    def update_knowledge(self, model_name: str, new_knowledge: Any) -> None:
    def delete_knowledge(self, model_name: str) -> None:
```
📌 **کاربرد**: ذخیره، بازیابی، به‌روزرسانی و حذف دانش مدل‌ها.

#### 🔹 `sharing_optimizer.py`
```python
class SharingOptimizer:
    def optimize_sharing(self, model_name: str, knowledge: Any) -> Any:
    def get_optimized_knowledge(self, model_name: str) -> Any:
```
📌 **کاربرد**: بهینه‌سازی فرآیند اشتراک دانش و کاهش سربار انتقال داده‌ها.

#### 🔹 `privacy_guard.py`
```python
class PrivacyGuard:
    def protect_privacy(self, knowledge: Dict[str, Any]) -> Dict[str, Any]:
    def add_sensitive_key(self, key: str) -> None:
    def remove_sensitive_key(self, key: str) -> None:
```
📌 **کاربرد**: حذف اطلاعات حساس از دانش قبل از اشتراک‌گذاری برای حفظ حریم خصوصی.

---

### **2️⃣ sync/** - هماهنگ‌سازی و کنترل نسخه دانش
#### 🔹 `sync_manager.py`
```python
class SyncManager:
    def update_sync_state(self, model_name: str, version: str) -> None:
    def get_sync_state(self, model_name: str) -> str:
    def is_synced(self, model_name: str, version: str) -> bool:
```
📌 **کاربرد**: مدیریت وضعیت همگام‌سازی مدل‌ها و نسخه‌های دانش.

#### 🔹 `conflict_resolver.py`
```python
class ConflictResolver:
    def detect_conflict(self, model_name: str, local_version: str, remote_version: str) -> bool:
    def resolve_conflict(self, model_name: str, resolution_strategy: str = "latest") -> str:
```
📌 **کاربرد**: تشخیص و حل تعارض بین نسخه‌های مختلف دانش.

#### 🔹 `version_controller.py`
```python
class VersionController:
    def add_version(self, model_name: str, version: str) -> None:
    def get_latest_version(self, model_name: str) -> str:
    def rollback_version(self, model_name: str, steps: int = 1) -> str:
```
📌 **کاربرد**: مدیریت نسخه‌های مختلف دانش و امکان بازگردانی نسخه‌های قبلی.

---

### **3️⃣ transfer/** - انتقال داده‌های فشرده و بررسی یکپارچگی
#### 🔹 `data_compressor.py`
```python
class DataCompressor:
    def compress_data(self, data: Any) -> bytes:
    def decompress_data(self, compressed_data: bytes) -> str:
```
📌 **کاربرد**: فشرده‌سازی و استخراج داده‌های اشتراکی برای کاهش حجم انتقال.

#### 🔹 `efficient_transfer.py`
```python
class EfficientTransfer:
    def prepare_data_for_transfer(self, data: Any) -> bytes:
    def receive_transferred_data(self, compressed_data: bytes) -> Any:
```
📌 **کاربرد**: انتقال بهینه دانش بین مدل‌ها با استفاده از فشرده‌سازی.

#### 🔹 `integrity_checker.py`
```python
class IntegrityChecker:
    def generate_checksum(self, data: Any) -> str:
    def verify_checksum(self, data: Any, expected_checksum: str) -> bool:
```
📌 **کاربرد**: بررسی صحت و یکپارچگی داده‌های منتقل‌شده با استفاده از هشینگ.

---

✅ **تمامی کلاس‌های ماژول `knowledge_sharing/` به صورت کامل پیاده‌سازی و مستند شدند.** 🎉





# 📌 مستندات ماژول `orchestration/` در Federation

## 📂 ساختار ماژول
```
federation/
├── orchestration/                     # هماهنگ‌سازی ارتباطات بین مدل‌ها
│   ├── coordinator/                   # هماهنگ‌کننده اجرای مدل‌ها و تخصیص منابع
│   │   ├── model_coordinator.py       # هماهنگ‌سازی بین مدل‌های هوش مصنوعی
│   │   ├── resource_coordinator.py    # مدیریت تخصیص منابع بین مدل‌ها
│   │   └── task_coordinator.py        # هماهنگ‌سازی وظایف بین مدل‌ها و توزیع پردازش
│   │
│   ├── monitor/                        # پایش عملکرد و سلامت پردازش‌ها
│   │   ├── health_monitor.py           # بررسی سلامت مدل‌ها و منابع
│   │   ├── performance_monitor.py      # پایش کارایی و عملکرد مدل‌ها
│   │   └── quality_monitor.py          # نظارت بر کیفیت پردازش و خروجی مدل‌ها
│   │
│   └── optimizer/                      # بهینه‌سازی فرآیند هماهنگ‌سازی
│       ├── orchestration_optimizer.py  # بهینه‌سازی هماهنگی بین مدل‌ها
│       ├── workflow_optimizer.py       # بهینه‌سازی روند اجرای پردازش‌ها
│       └── timing_optimizer.py         # بهینه‌سازی زمان‌بندی اجرای وظایف
```

---

## 📌 جزئیات کلاس‌ها و متدهای ماژول `orchestration/`

### **1️⃣ coordinator/** - هماهنگ‌کننده اجرای مدل‌ها و تخصیص منابع
#### 🔹 `model_coordinator.py`
```python
class ModelCoordinator:
    def register_model(self, model_name: str) -> None:
    def assign_task(self, model_name: str, task: Dict[str, Any]) -> bool:
    def release_model(self, model_name: str) -> None:
    def get_available_models(self) -> List[str]:
```
📌 **کاربرد**: مدیریت هماهنگی بین مدل‌های مختلف برای تخصیص وظایف و جلوگیری از تداخل پردازشی.

#### 🔹 `resource_coordinator.py`
```python
class ResourceCoordinator:
    def allocate_resources(self, model_name: str, cpu: float, memory: float) -> bool:
    def release_resources(self, model_name: str) -> bool:
    def get_allocated_resources(self, model_name: str) -> Dict[str, float]:
```
📌 **کاربرد**: مدیریت تخصیص منابع پردازشی بین مدل‌ها و بهینه‌سازی مصرف منابع.

#### 🔹 `task_coordinator.py`
```python
class TaskCoordinator:
    def add_task(self, task: Dict[str, Any]) -> None:
    def assign_task(self) -> Dict[str, Any]:
    def get_pending_tasks(self) -> List[Dict[str, Any]]:
```
📌 **کاربرد**: مدیریت و توزیع وظایف بین مدل‌ها برای بهینه‌سازی پردازش.

---

### **2️⃣ monitor/** - پایش عملکرد و سلامت پردازش‌ها
#### 🔹 `health_monitor.py`
```python
class HealthMonitor:
    def update_health_status(self, model_name: str, status: str) -> None:
    def get_health_status(self, model_name: str) -> str:
    def get_all_health_statuses(self) -> Dict[str, str]:
```
📌 **کاربرد**: بررسی سلامت مدل‌ها و منابع پردازشی و شناسایی مشکلات احتمالی.

#### 🔹 `performance_monitor.py`
```python
class PerformanceMonitor:
    def update_performance(self, model_name: str, latency: float, throughput: float) -> None:
    def get_performance(self, model_name: str) -> Dict[str, float]:
    def get_all_performance_data(self) -> Dict[str, Dict[str, float]]:
```
📌 **کاربرد**: پایش کارایی مدل‌ها از جمله تأخیر پردازش و نرخ پردازش.

#### 🔹 `quality_monitor.py`
```python
class QualityMonitor:
    def update_quality_metrics(self, model_name: str, accuracy: float, precision: float, recall: float) -> None:
    def get_quality_metrics(self, model_name: str) -> Dict[str, float]:
    def get_all_quality_metrics(self) -> Dict[str, Dict[str, float]]:
```
📌 **کاربرد**: نظارت بر کیفیت پردازش و ارزیابی دقت مدل‌ها.

---

### **3️⃣ optimizer/** - بهینه‌سازی فرآیند هماهنگ‌سازی
#### 🔹 `orchestration_optimizer.py`
```python
class OrchestrationOptimizer:
    def analyze_workflow(self, model_name: str, execution_time: float, resource_usage: float) -> None:
    def get_optimization_data(self, model_name: str) -> Dict[str, float]:
    def suggest_improvement(self, model_name: str) -> str:
```
📌 **کاربرد**: تحلیل روند اجرای مدل‌ها و ارائه پیشنهادات برای بهینه‌سازی فرآیندها.

#### 🔹 `workflow_optimizer.py`
```python
class WorkflowOptimizer:
    def log_execution(self, model_name: str, task_type: str, duration: float) -> None:
    def get_execution_history(self) -> List[Dict[str, Any]]:
    def suggest_workflow_improvement(self) -> str:
```
📌 **کاربرد**: بهینه‌سازی روند اجرای پردازش‌ها برای افزایش بهره‌وری سیستم.

#### 🔹 `timing_optimizer.py`
```python
class TimingOptimizer:
    def log_task_execution(self, model_name: str, task_type: str) -> None:
    def analyze_execution_timing(self) -> str:
```
📌 **کاربرد**: بهینه‌سازی زمان‌بندی اجرای وظایف برای کاهش تأخیر و افزایش کارایی.

---

✅ **تمامی کلاس‌های ماژول `orchestration/` به صورت کامل پیاده‌سازی و مستند شدند.** 🎉





# 📌 مستندات ماژول `learning/` در Federation

## 📂 ساختار ماژول
```
federation/
├── learning/                         # مدیریت یادگیری فدراسیونی بین مدل‌ها
│   ├── federation/                    # یادگیری فدراسیونی و تجمیع مدل‌ها
│   │   ├── federated_learner.py       # هماهنگ‌کننده یادگیری فدراسیونی
│   │   ├── model_aggregator.py        # تجمیع اطلاعات و وزن‌های مدل‌ها
│   │   └── learning_optimizer.py      # بهینه‌سازی فرآیند یادگیری
│   │
│   ├── privacy/                        # حفظ امنیت و حریم خصوصی داده‌ها
│   │   ├── privacy_preserving.py       # یادگیری با حریم خصوصی حفظ‌شده
│   │   ├── data_anonymizer.py          # ناشناس‌سازی داده‌ها قبل از پردازش
│   │   └── security_manager.py         # مدیریت امنیتی برای جلوگیری از نشت اطلاعات
│   │
│   └── adaptation/                     # هماهنگ‌سازی دانش بین مدل‌های مختلف
│       ├── model_adapter.py            # سازگارسازی مدل‌ها برای تبادل دانش
│       ├── knowledge_adapter.py        # تنظیم و یکپارچه‌سازی دانش بین مدل‌ها
│       └── strategy_adapter.py         # بهینه‌سازی استراتژی‌های یادگیری بین مدل‌ها
```

---

## 📌 جزئیات کلاس‌ها و متدهای ماژول `learning/`

### **1️⃣ federation/** - مدیریت یادگیری فدراسیونی و تجمیع مدل‌ها
#### 🔹 `federated_learner.py`
```python
class FederatedLearner:
    def collect_model_update(self, model_name: str, update: List[float]) -> None:
    def get_updates(self) -> Dict[str, List[List[float]]]:
    def clear_updates(self) -> None:
```
📌 **کاربرد**: مدیریت فرآیند یادگیری فدراسیونی و هماهنگی بین مدل‌ها برای به‌روزرسانی وزن‌ها.

#### 🔹 `model_aggregator.py`
```python
class ModelAggregator:
    def aggregate_updates(self, model_updates: Dict[str, List[float]]) -> List[float]:
    def get_aggregated_weights(self) -> List[float]:
```
📌 **کاربرد**: تجمیع به‌روزرسانی‌های مدل‌های مختلف و ایجاد وزن‌های جدید مدل.

#### 🔹 `learning_optimizer.py`
```python
class LearningOptimizer:
    def adjust_learning_rate(self, previous_losses: List[float]) -> float:
    def get_learning_rates(self) -> List[float]:
```
📌 **کاربرد**: بهینه‌سازی فرآیند یادگیری و تنظیم نرخ یادگیری مدل‌ها.

---

### **2️⃣ privacy/** - حفظ امنیت و حریم خصوصی داده‌ها
#### 🔹 `privacy_preserving.py`
```python
class PrivacyPreserving:
    def apply_differential_privacy(self, model_weights: List[float]) -> List[float]:
    def set_noise_level(self, new_noise_level: float) -> None:
```
📌 **کاربرد**: اعمال روش‌های حفظ حریم خصوصی مانند **Differential Privacy** برای جلوگیری از افشای داده‌های خام.

#### 🔹 `data_anonymizer.py`
```python
class DataAnonymizer:
    def anonymize_data(self, data: Dict[str, str]) -> Dict[str, str]:
    def set_salt(self, new_salt: str) -> None:
```
📌 **کاربرد**: ناشناس‌سازی داده‌های حساس کاربران با استفاده از هشینگ و `salt`.

#### 🔹 `security_manager.py`
```python
class SecurityManager:
    def generate_access_key(self, user_id: str) -> str:
    def verify_access_key(self, user_id: str, access_key: str) -> bool:
    def revoke_access(self, user_id: str) -> None:
```
📌 **کاربرد**: مدیریت احراز هویت کاربران و جلوگیری از نشت داده‌ها.

---

### **3️⃣ adaptation/** - هماهنگ‌سازی دانش بین مدل‌های مختلف
#### 🔹 `model_adapter.py`
```python
class ModelAdapter:
    def register_model(self, model_name: str, compatible_models: List[str]) -> None:
    def get_compatible_models(self, model_name: str) -> List[str]:
    def is_compatible(self, source_model: str, target_model: str) -> bool:
```
📌 **کاربرد**: سازگارسازی مدل‌ها برای تبادل دانش و استفاده از یادگیری فدراسیونی.

#### 🔹 `knowledge_adapter.py`
```python
class KnowledgeAdapter:
    def register_knowledge(self, model_name: str, knowledge: Any) -> None:
    def get_knowledge(self, model_name: str) -> Any:
    def transfer_knowledge(self, source_model: str, target_model: str) -> bool:
```
📌 **کاربرد**: تنظیم و یکپارچه‌سازی دانش بین مدل‌های مختلف برای یادگیری بهتر.

#### 🔹 `strategy_adapter.py`
```python
class StrategyAdapter:
    def register_strategy(self, model_name: str, strategy: str) -> None:
    def get_strategy(self, model_name: str) -> str:
    def update_strategy(self, model_name: str, new_strategy: str) -> None:
```
📌 **کاربرد**: بهینه‌سازی استراتژی‌های یادگیری برای بهبود عملکرد مدل‌ها.

---

✅ **تمامی کلاس‌های ماژول `learning/` به صورت کامل پیاده‌سازی و مستند شدند.** 🎉



# 📌 مستندات ماژول `metrics/` در Federation

## 📂 ساختار ماژول
```
federation/
├── metrics/                          # پایش و ارزیابی عملکرد سیستم
│   ├── collectors/                    # جمع‌آوری داده‌های متریک
│   │   ├── performance_collector.py   # جمع‌آوری متریک‌های مربوط به کارایی مدل‌ها
│   │   ├── efficiency_collector.py    # جمع‌آوری متریک‌های بهره‌وری منابع
│   │   └── quality_collector.py       # جمع‌آوری متریک‌های کیفیت خروجی مدل‌ها
│   │
│   ├── analyzers/                     # تحلیل داده‌های متریک و شناسایی الگوها
│   │   ├── metric_analyzer.py         # تحلیل متریک‌های کل سیستم و تشخیص مشکلات
│   │   ├── pattern_detector.py        # شناسایی الگوها در داده‌های عملکردی
│   │   └── trend_analyzer.py          # بررسی روندهای عملکرد مدل‌ها در طول زمان
│   │
│   └── optimizers/                    # بهینه‌سازی بر اساس متریک‌های تحلیل‌شده
│       ├── metric_optimizer.py        # بهینه‌سازی متریک‌های عملکردی سیستم
│       ├── alert_manager.py           # مدیریت هشدارهای سیستمی برای مشکلات متریک‌ها
│       └── report_generator.py        # تولید گزارش‌های تحلیل متریک‌ها
```

---

## 📌 جزئیات کلاس‌ها و متدهای ماژول `metrics/`

### **1️⃣ collectors/** - جمع‌آوری داده‌های متریک
#### 🔹 `performance_collector.py`
```python
class PerformanceCollector:
    def collect_metrics(self, model_name: str, accuracy: float, latency: float, throughput: float) -> None:
    def get_metrics(self, model_name: str) -> Dict[str, float]:
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
```
📌 **کاربرد**: جمع‌آوری متریک‌های عملکردی شامل **دقت، تأخیر و نرخ پردازش مدل‌ها**.

#### 🔹 `efficiency_collector.py`
```python
class EfficiencyCollector:
    def collect_metrics(self, model_name: str, cpu_usage: float, memory_usage: float, gpu_usage: float) -> None:
    def get_metrics(self, model_name: str) -> Dict[str, float]:
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
```
📌 **کاربرد**: جمع‌آوری متریک‌های مربوط به **مصرف CPU، حافظه و GPU مدل‌ها**.

#### 🔹 `quality_collector.py`
```python
class QualityCollector:
    def collect_metrics(self, model_name: str, accuracy: float, precision: float, recall: float) -> None:
    def get_metrics(self, model_name: str) -> Dict[str, float]:
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
```
📌 **کاربرد**: جمع‌آوری متریک‌های **کیفیت خروجی مدل‌ها شامل دقت، صحت و بازیابی**.

---

### **2️⃣ analyzers/** - تحلیل داده‌های متریک و شناسایی الگوها
#### 🔹 `metric_analyzer.py`
```python
class MetricAnalyzer:
    def analyze_metrics(self, model_name: str, metrics: Dict[str, float]) -> str:
    def get_analysis(self, model_name: str) -> Dict[str, Any]:
```
📌 **کاربرد**: تحلیل متریک‌های کل سیستم و تشخیص مشکلات عملکردی.

#### 🔹 `pattern_detector.py`
```python
class PatternDetector:
    def record_metrics(self, model_name: str, metric_values: List[float]) -> None:
    def detect_pattern(self, model_name: str) -> str:
```
📌 **کاربرد**: شناسایی **الگوهای پرتکرار در متریک‌های عملکردی مدل‌ها**.

#### 🔹 `trend_analyzer.py`
```python
class TrendAnalyzer:
    def record_metrics(self, model_name: str, metric_value: float) -> None:
    def analyze_trend(self, model_name: str) -> str:
```
📌 **کاربرد**: بررسی **روندهای بلندمدت تغییرات متریک‌ها** و تحلیل روند عملکرد مدل‌ها.

---

### **3️⃣ optimizers/** - بهینه‌سازی بر اساس متریک‌های تحلیل‌شده
#### 🔹 `metric_optimizer.py`
```python
class MetricOptimizer:
    def optimize_metrics(self, model_name: str, metrics: Dict[str, float]) -> str:
    def get_optimization_data(self, model_name: str) -> Dict[str, Any]:
```
📌 **کاربرد**: ارائه پیشنهادات برای **بهبود عملکرد مدل‌ها بر اساس متریک‌های آن‌ها**.

#### 🔹 `alert_manager.py`
```python
class AlertManager:
    def generate_alert(self, model_name: str, metric: str, value: float, threshold: float) -> str:
    def get_alerts(self) -> Dict[str, str]:
    def clear_alerts(self) -> None:
```
📌 **کاربرد**: **مدیریت هشدارهای سیستمی** برای مشکلات متریک‌ها و اعلام وضعیت بحرانی.

#### 🔹 `report_generator.py`
```python
class ReportGenerator:
    def generate_report(self, model_name: str, metrics: Dict[str, float]) -> str:
    def get_report(self, model_name: str) -> str:
    def get_all_reports(self) -> Dict[str, str]:
```
📌 **کاربرد**: **تولید گزارش‌های تحلیل متریک‌ها** برای نظارت و بهینه‌سازی سیستم.

---

✅ **تمامی کلاس‌های ماژول `metrics/` به صورت کامل پیاده‌سازی و مستند شدند.** 🎉



