### **🚀 مستندات ماژول `intelligence` **  
📌 **معرفی ماژول و هدف آن**  

---

### **۱️⃣ معرفی کلی ماژول Intelligence**
ماژول `intelligence` در پروژه **AI Data** نقش اصلی در **تحلیل، بهینه‌سازی، و زمان‌بندی پردازش‌های داده‌ای** را ایفا می‌کند. این ماژول شامل مجموعه‌ای از ابزارهای **تشخیص الگو، پیش‌بینی بار پردازشی، توزیع منابع، و مدیریت وابستگی‌های پردازشی** است که در کنار هم باعث **افزایش کارایی سیستم پردازش داده‌ها** می‌شوند.  

### **۲️⃣ محل قرارگیری ماژول در ساختار پروژه**
دایرکتوری این ماژول در ساختار پروژه در مسیر زیر قرار دارد:
```
ai/data/intelligence/
```
تمام فایل‌ها و زیرماژول‌های مربوط به پردازش هوشمند داده‌ها در این دایرکتوری قرار گرفته‌اند.

---

### **۳️⃣ ساختار پوشه‌بندی ماژول Intelligence**
```
intelligence/
├── analyzer/                   
│   ├── __init__.py                # مدیریت ماژول Analyzer
│   ├── bottleneck_detector.py      # شناسایی گلوگاه‌های پردازشی
│   ├── efficiency_monitor.py       # پایش عملکرد سیستم
│   ├── load_predictor.py           # پیش‌بینی بار پردازشی آینده
│   ├── pattern_detector.py         # تشخیص الگوهای داده‌ای
│   ├── performance_analyzer.py     # تحلیل کارایی پردازش‌ها
│   ├── quality_checker.py          # بررسی کیفیت داده‌های پردازشی
│
├── optimizer/
│   ├── __init__.py                 # مدیریت ماژول Optimizer
│   ├── dependency_manager.py        # مدیریت وابستگی پردازش‌ها
│   ├── memory_optimizer.py          # بهینه‌سازی مصرف حافظه
│   ├── resource_balancer.py         # مدیریت توزیع منابع پردازشی
│   ├── stream_optimizer.py          # بهینه‌سازی پردازش جریانی
│   ├── task_scheduler.py            # زمان‌بندی پردازش‌های داده‌ای
│   ├── throughput_optimizer.py      # بهینه‌سازی توان عملیاتی پردازش‌ها
│   ├── workload_balancer.py         # توزیع بار پردازشی بین منابع مختلف
│
├── scheduler/
│   ├── __init__.py                  # مدیریت ماژول Scheduler
│   ├── priority_manager.py           # مدیریت اولویت پردازش‌ها
│
├── __init__.py                       # مدیریت ماژول Intelligence
```

---

## **📌 توضیح ساختار کلی ماژول Intelligence**
🔹 **ماژول `analyzer/`** → شامل ابزارهای **تحلیل داده‌ها و تشخیص الگوهای پردازشی** است. این ماژول مسئولیت **تشخیص گلوگاه‌های پردازشی، پایش کیفیت داده‌ها، و پیش‌بینی روند بار پردازشی آینده** را دارد.  
🔹 **ماژول `optimizer/`** → شامل ابزارهای **بهینه‌سازی منابع و توزیع بار پردازشی** است. این ماژول وظیفه **افزایش توان عملیاتی پردازش‌ها، مدیریت وابستگی‌ها، و تخصیص بهینه حافظه و منابع پردازشی** را دارد.  
🔹 **ماژول `scheduler/`** → شامل ابزارهای **مدیریت اولویت و زمان‌بندی پردازش‌های داده‌ای** است. این ماژول پردازش‌های داده را بر اساس **اولویت، میزان مصرف منابع، و وضعیت سیستم** زمان‌بندی می‌کند.  

---

### **🎯 نتیجه‌گیری**
✅ **ماژول `intelligence` شامل ابزارهای تحلیل، بهینه‌سازی، و زمان‌بندی پردازش داده‌ها است.**  
✅ **ساختار این ماژول طوری طراحی شده که پردازش‌های سنگین را کارآمدتر مدیریت کند و بهینه‌سازی منابع را تضمین کند.**  
✅ **تمام ابزارهای مورد نیاز برای افزایش کارایی سیستم پردازش داده‌ها در این ماژول قرار داده شده‌اند.**  

---

### **🚀 مستندات ماژول `analyzer` - بخش دوم**  
📌 **توضیح کامل عملکرد `analyzer/`، کلاس‌ها، متدها، و مکانیسم‌های مورد استفاده**  

---

## **📌  بررسی ماژول `analyzer/`**
📌 **ماژول `analyzer` چیست و چه کاربردی دارد؟**  
ماژول `analyzer/` یکی از مهم‌ترین بخش‌های `intelligence/` است و شامل ابزارهایی برای **تحلیل داده‌های پردازشی، تشخیص الگوها، و بررسی کیفیت پردازش‌ها** می‌باشد. این ماژول نقش مهمی در **بهینه‌سازی عملکرد سیستم پردازش داده‌ها** دارد.

📌 **ویژگی‌های کلیدی این ماژول:**  
- **تحلیل عملکرد سیستم پردازشی و شناسایی گلوگاه‌ها**  
- **پایش کیفیت داده‌ها برای جلوگیری از ورود داده‌های نامعتبر**  
- **پیش‌بینی تغییرات بار پردازشی برای جلوگیری از سربار ناگهانی**  
- **تشخیص الگوهای پردازشی برای افزایش کارایی مدل‌های یادگیری ماشین**  

---

### **📂 ساختار ماژول `analyzer/`**
```
analyzer/
├── __init__.py                # مدیریت ماژول Analyzer
├── bottleneck_detector.py      # شناسایی گلوگاه‌های پردازشی
├── efficiency_monitor.py       # پایش عملکرد سیستم
├── load_predictor.py           # پیش‌بینی بار پردازشی آینده
├── pattern_detector.py         # تشخیص الگوهای داده‌ای
├── performance_analyzer.py     # تحلیل کارایی پردازش‌ها
└── quality_checker.py          # بررسی کیفیت داده‌های پردازشی
```

---

## **📌 بررسی تک‌تک فایل‌های `analyzer/`**
### **۱️⃣ `bottleneck_detector.py` - شناسایی گلوگاه‌های پردازشی**
📌 **هدف:**  
این ماژول مسئول **تحلیل فرآیندهای پردازشی و شناسایی نقاطی است که باعث کاهش سرعت پردازش می‌شوند**.  

📌 **مکانیسم‌های کلیدی:**  
- **تحلیل تأخیر پردازشی** → بررسی تأخیر پردازش‌های داده‌ای  
- **بررسی میزان استفاده از منابع** → شناسایی پردازش‌هایی که بیش از حد از CPU یا حافظه استفاده می‌کنند  
- **ارائه پیشنهاد برای رفع گلوگاه‌ها**  

📌 **ساختار کلی کلاس:**  
```python
class BottleneckDetector:
    def detect_bottlenecks(self, metrics_data):
        """
        تحلیل داده‌های عملکردی برای شناسایی گلوگاه‌ها
        """
```

---

### **۲️⃣ `efficiency_monitor.py` - پایش عملکرد سیستم**
📌 **هدف:**  
این ماژول داده‌های عملکردی را پایش می‌کند تا میزان کارایی سیستم را **در زمان واقعی تحلیل کند.**  

📌 **مکانیسم‌های کلیدی:**  
- **مقایسه کارایی با داده‌های تاریخی**  
- **محاسبه میزان استفاده بهینه از منابع پردازشی**  

📌 **ساختار کلی کلاس:**  
```python
class EfficiencyMonitor:
    def monitor_efficiency(self, system_metrics):
        """
        پایش داده‌های عملکردی سیستم برای بهینه‌سازی کارایی
        """
```

---

### **۳️⃣ `load_predictor.py` - پیش‌بینی بار پردازشی آینده**
📌 **هدف:**  
این ماژول با بررسی الگوهای گذشته، میزان **بار پردازشی آینده را پیش‌بینی می‌کند.**  

📌 **مکانیسم‌های کلیدی:**  
- **مدل‌سازی سری‌های زمانی برای پیش‌بینی تغییرات بار پردازشی**  
- **تشخیص روند‌های افزایش یا کاهش بار پردازشی**  
- **ارائه پیشنهاد برای مقیاس‌گذاری منابع**  

📌 **ساختار کلی کلاس:**  
```python
class LoadPredictor:
    def predict_load(self, historical_data):
        """
        پیش‌بینی میزان بار پردازشی بر اساس داده‌های قبلی
        """
```

---

### **۴️⃣ `pattern_detector.py` - تشخیص الگوهای داده‌ای**
📌 **هدف:**  
این ماژول داده‌های ورودی را تحلیل کرده و **الگوهای تکراری در پردازش‌ها را شناسایی می‌کند.**  

📌 **مکانیسم‌های کلیدی:**  
- **تحلیل سری‌های زمانی برای تشخیص الگوهای پرتکرار**  
- **مدل‌سازی تغییرات ناگهانی در پردازش داده‌ها**  
- **شناسایی پردازش‌های غیرعادی**  

📌 **ساختار کلی کلاس:**  
```python
class PatternDetector:
    def detect_patterns(self, data_stream):
        """
        تحلیل داده‌های ورودی و تشخیص الگوهای پرتکرار
        """
```

---

### **۵️⃣ `performance_analyzer.py` - تحلیل کارایی پردازش‌ها**
📌 **هدف:**  
این ماژول **میزان کارایی پردازش‌های داده‌ای را بررسی و تحلیل می‌کند.**  

📌 **مکانیسم‌های کلیدی:**  
- **بررسی میزان مصرف منابع پردازشی توسط هر فرآیند**  
- **محاسبه کارایی کلی سیستم بر اساس معیارهای استاندارد**  
- **ارائه گزارش برای بهینه‌سازی پردازش‌های داده‌ای**  

📌 **ساختار کلی کلاس:**  
```python
class PerformanceAnalyzer:
    def analyze_performance(self, process_metrics):
        """
        بررسی میزان کارایی پردازش‌های داده‌ای
        """
```

---

### **۶️⃣ `quality_checker.py` - بررسی کیفیت داده‌های پردازشی**
📌 **هدف:**  
این ماژول مسئول **تحلیل کیفیت داده‌های پردازشی و تشخیص داده‌های نامعتبر یا ناسازگار است.**  

📌 **مکانیسم‌های کلیدی:**  
- **شناسایی داده‌های نادرست و نامعتبر در پردازش‌ها**  
- **تشخیص داده‌های ناقص یا ناسازگار**  
- **فیلتر کردن داده‌های غیرقابل استفاده قبل از پردازش**  

📌 **ساختار کلی کلاس:**  
```python
class QualityChecker:
    def check_data_quality(self, dataset):
        """
        بررسی کیفیت داده‌های پردازشی و تشخیص داده‌های ناسازگار
        """
```

---

### **🎯 نتیجه‌گیری**
✅ **ماژول `analyzer/` نقش کلیدی در تحلیل و نظارت بر پردازش‌های داده‌ای دارد.**  
✅ **هر فایل شامل ابزاری برای تحلیل یکی از جنبه‌های کلیدی پردازش داده‌ها است، از جمله: تشخیص گلوگاه‌ها، پایش کارایی، تشخیص الگوها، و پیش‌بینی بار پردازشی آینده.**  
✅ **این ماژول پایه‌ای برای بهینه‌سازی عملکرد سیستم و افزایش بهره‌وری است.**  

---

### **🚀 مستندات ماژول `optimizer`**  
📌 **توضیح کامل عملکرد `optimizer/`، کلاس‌ها، متدها، و مکانیسم‌های مورد استفاده**  

---

## **📌 بخش سوم: بررسی ماژول `optimizer/`**
📌 **ماژول `optimizer` چیست و چه کاربردی دارد؟**  
ماژول `optimizer/` شامل ابزارهایی برای **بهینه‌سازی منابع، مدیریت تخصیص حافظه، توزیع پردازش‌ها، و بهبود کارایی پردازش‌های داده‌ای** است. این ماژول وظیفه دارد **مطمئن شود که پردازش‌ها به طور بهینه از منابع استفاده می‌کنند و سیستم دچار سربار پردازشی نمی‌شود.**  

📌 **ویژگی‌های کلیدی این ماژول:**  
- **مدیریت تخصیص حافظه و جلوگیری از هدررفت منابع**  
- **توزیع پردازش‌ها بین منابع پردازشی مختلف برای کاهش گلوگاه‌ها**  
- **مدیریت وابستگی‌ها بین پردازش‌ها برای جلوگیری از تأخیرهای غیرضروری**  
- **زمان‌بندی و بهینه‌سازی اولویت پردازش‌ها برای اجرای بهینه**  

---

### **📂 ساختار ماژول `optimizer/`**
```
optimizer/
├── __init__.py                 # مدیریت ماژول Optimizer
├── dependency_manager.py        # مدیریت وابستگی پردازش‌ها
├── memory_optimizer.py          # بهینه‌سازی مصرف حافظه
├── resource_balancer.py         # مدیریت توزیع منابع پردازشی
├── stream_optimizer.py          # بهینه‌سازی پردازش جریانی
├── task_scheduler.py            # زمان‌بندی پردازش‌های داده‌ای
├── throughput_optimizer.py      # بهینه‌سازی توان عملیاتی پردازش‌ها
└── workload_balancer.py         # توزیع بار پردازشی بین منابع مختلف
```

---

## **📌 بررسی تک‌تک فایل‌های `optimizer/`**
### **۱️⃣ `dependency_manager.py` - مدیریت وابستگی پردازش‌ها**
📌 **هدف:**  
این ماژول مسئول **مدیریت وابستگی‌های بین پردازش‌ها** است تا از **تأخیرهای غیرضروری یا مشکلات اجرای پردازش‌ها** جلوگیری کند.  

📌 **مکانیسم‌های کلیدی:**  
- **مدیریت وابستگی بین پردازش‌ها با استفاده از گراف جهت‌دار**  
- **تشخیص حلقه‌های وابستگی و حل آن‌ها**  

📌 **ساختار کلی کلاس:**  
```python
class DependencyManager:
    def add_dependency(self, task_id, depends_on):
        """
        اضافه کردن وابستگی بین پردازش‌ها
        """
```

---

### **۲️⃣ `memory_optimizer.py` - بهینه‌سازی مصرف حافظه**
📌 **هدف:**  
این ماژول **مصرف حافظه (RAM) را پایش و بهینه‌سازی می‌کند** تا از سرریز حافظه جلوگیری شود.  

📌 **مکانیسم‌های کلیدی:**  
- **تحلیل میزان استفاده از حافظه و پیشنهاد بهینه‌سازی**  
- **کاهش حافظه مصرفی از طریق فشرده‌سازی داده‌ها**  

📌 **ساختار کلی کلاس:**  
```python
class MemoryOptimizer:
    def optimize_memory(self):
        """
        بهینه‌سازی مصرف حافظه در پردازش‌های داده‌ای
        """
```

---

### **۳️⃣ `resource_balancer.py` - مدیریت توزیع منابع پردازشی**
📌 **هدف:**  
این ماژول پردازش‌های سنگین را **بین منابع پردازشی مختلف (CPU، RAM، I/O) توزیع می‌کند** تا از گلوگاه‌های پردازشی جلوگیری شود.  

📌 **مکانیسم‌های کلیدی:**  
- **تحلیل مصرف CPU، RAM و I/O و توزیع مجدد پردازش‌ها**  
- **کاهش مصرف بیش از حد یک منبع خاص**  

📌 **ساختار کلی کلاس:**  
```python
class ResourceBalancer:
    def balance_resources(self):
        """
        توزیع پردازش‌ها بین منابع مختلف برای جلوگیری از گلوگاه‌های پردازشی
        """
```

---

### **۴️⃣ `stream_optimizer.py` - بهینه‌سازی پردازش جریانی**
📌 **هدف:**  
این ماژول مسئول **مدیریت و بهینه‌سازی پردازش داده‌های جریانی (Stream Processing)** است.  

📌 **مکانیسم‌های کلیدی:**  
- **کاهش تأخیر پردازش جریانی**  
- **مدیریت حجم داده‌های ورودی و خروجی برای جلوگیری از سربار پردازشی**  

📌 **ساختار کلی کلاس:**  
```python
class StreamOptimizer:
    def optimize_stream(self, data_stream):
        """
        بهینه‌سازی پردازش جریانی داده‌ها
        """
```

---

### **۵️⃣ `task_scheduler.py` - زمان‌بندی پردازش‌های داده‌ای**
📌 **هدف:**  
این ماژول پردازش‌ها را **بر اساس اولویت و وضعیت منابع سیستم، زمان‌بندی می‌کند.**  

📌 **مکانیسم‌های کلیدی:**  
- **بررسی میزان مصرف منابع و اولویت‌بندی پردازش‌ها**  
- **مدیریت صف پردازش‌ها با الگوریتم‌های زمان‌بندی مانند Priority Scheduling**  

📌 **ساختار کلی کلاس:**  
```python
class TaskScheduler:
    def schedule_tasks(self, tasks):
        """
        زمان‌بندی پردازش‌های داده‌ای بر اساس میزان مصرف منابع
        """
```

---

### **۶️⃣ `throughput_optimizer.py` - بهینه‌سازی توان عملیاتی پردازش‌ها**
📌 **هدف:**  
این ماژول وظیفه **افزایش نرخ پردازش داده‌ها و کاهش زمان پردازش کلی سیستم** را دارد.  

📌 **مکانیسم‌های کلیدی:**  
- **تحلیل نرخ پردازش و پیشنهاد بهینه‌سازی منابع**  
- **مدیریت پردازش‌های همزمان برای افزایش کارایی**  

📌 **ساختار کلی کلاس:**  
```python
class ThroughputOptimizer:
    def optimize_throughput(self, process_metrics):
        """
        افزایش نرخ پردازش داده‌ها و کاهش تأخیر
        """
```

---

### **۷️⃣ `workload_balancer.py` - توزیع بار پردازشی بین منابع مختلف**
📌 **هدف:**  
این ماژول بار پردازشی بین پردازش‌ها و منابع مختلف را **بهینه‌سازی و توزیع می‌کند.**  

📌 **مکانیسم‌های کلیدی:**  
- **انتقال پردازش‌های سنگین به منابع کمتر استفاده‌شده**  
- **مدیریت میزان مصرف منابع برای جلوگیری از سرریز پردازشی**  

📌 **ساختار کلی کلاس:**  
```python
class WorkloadBalancer:
    def balance_workload(self, processes):
        """
        توزیع بار پردازشی بین منابع مختلف برای افزایش کارایی
        """
```

---

### **🎯 نتیجه‌گیری**
✅ **ماژول `optimizer/` نقش کلیدی در بهینه‌سازی منابع و مدیریت پردازش‌های داده‌ای دارد.**  
✅ **هر فایل شامل ابزاری برای یکی از جنبه‌های بهینه‌سازی پردازش، از جمله تخصیص حافظه، توزیع منابع، و زمان‌بندی پردازش‌ها است.**  
✅ **این ماژول کمک می‌کند تا سیستم از منابع خود به‌طور بهینه استفاده کند و از ایجاد گلوگاه‌های پردازشی جلوگیری شود.**  

---

### **🚀 مستندات ماژول `scheduler` - بخش چهارم**  
📌 **توضیح کامل عملکرد `scheduler/`، کلاس‌ها، متدها، و نحوه استفاده از این ماژول**  

---

## **📌 بخش چهارم: بررسی ماژول `scheduler/`**
📌 **ماژول `scheduler` چیست و چه کاربردی دارد؟**  
ماژول `scheduler/` شامل ابزارهایی برای **مدیریت زمان‌بندی و اولویت پردازش‌های داده‌ای** است. این ماژول تعیین می‌کند که **کدام پردازش‌ها زودتر اجرا شوند و منابع بیشتری دریافت کنند** تا سیستم بهینه‌تر عمل کند.  

📌 **ویژگی‌های کلیدی این ماژول:**  
- **زمان‌بندی پردازش‌ها بر اساس اولویت و منابع موجود**  
- **مدیریت اولویت پردازش‌های داده‌ای برای جلوگیری از تأخیرهای غیرضروری**  
- **افزایش سرعت پردازش داده‌ها با استفاده از تکنیک‌های مدیریت صف پردازش‌ها**  

---

### **📂 ساختار ماژول `scheduler/`**
```
scheduler/
├── __init__.py             # مدیریت ماژول Scheduler
└── priority_manager.py     # مدیریت اولویت پردازش‌ها
```

---

## **📌 بررسی فایل‌های `scheduler/`**
### **۱️⃣ `priority_manager.py` - مدیریت اولویت پردازش‌ها**
📌 **هدف:**  
این ماژول پردازش‌های داده‌ای را **بر اساس اهمیت و نیازمندی‌های پردازشی آن‌ها، اولویت‌بندی می‌کند.**  

📌 **مکانیسم‌های کلیدی:**  
- **بررسی میزان مصرف منابع و اهمیت پردازش‌ها**  
- **مدیریت صف پردازش‌ها با الگوریتم‌های زمان‌بندی مانند `Priority Queue`**  

📌 **ساختار کلی کلاس:**  
```python
import heapq

class PriorityManager:
    def __init__(self):
        self.priority_queue = []  # استفاده از heap برای مدیریت پردازش‌ها بر اساس اولویت

    def assign_priorities(self, tasks):
        """
        تعیین اولویت پردازش‌های داده‌ای.
        :param tasks: لیستی از پردازش‌های داده‌ای که باید اجرا شوند.
        :return: لیستی از پردازش‌ها به ترتیب اولویت.
        """
        if not tasks:
            return []

        for task in tasks:
            priority = task.get("priority", 5)  # مقدار پیش‌فرض اولویت (۱=بالاترین، ۱۰=کمترین)
            heapq.heappush(self.priority_queue, (priority, task))

        prioritized_tasks = []
        while self.priority_queue:
            _, task = heapq.heappop(self.priority_queue)
            prioritized_tasks.append(task)

        return prioritized_tasks
```

📌 **توضیح مکانیسم:**  
✅ **از `heapq` برای مدیریت صف پردازش‌ها بر اساس اولویت استفاده شده است.**  
✅ **اگر پردازشی نیاز به منابع بیشتری داشته باشد، اولویت کمتری می‌گیرد.**  
✅ **پردازش‌های مهم‌تر زودتر اجرا می‌شوند تا از تأخیرهای غیرضروری جلوگیری شود.**  

---

## **📌 نحوه استفاده از ماژول `scheduler/` در سایر بخش‌های پروژه**
📌 **ماژول `scheduler` کجا استفاده می‌شود؟**  
- **در ماژول `optimizer/`** برای **زمان‌بندی پردازش‌ها در `task_scheduler.py`**  
- **در ماژول `intelligence/`** برای **مدیریت پردازش‌های تحلیل داده در `performance_analyzer.py`**  
- **در ماژول `core/`** برای **مدیریت صف پردازش‌ها و بهینه‌سازی پردازش‌های در حال اجرا**  

📌 **نحوه استفاده از `PriorityManager` در سایر ماژول‌ها:**  
```python
from intelligence.scheduler.priority_manager import PriorityManager

tasks = [
    {"id": 1, "priority": 2, "cpu_demand": 0.6},
    {"id": 2, "priority": 1, "cpu_demand": 0.8},
    {"id": 3, "priority": 3, "cpu_demand": 0.4},
]

scheduler = PriorityManager()
ordered_tasks = scheduler.assign_priorities(tasks)

print("Execution Order:", ordered_tasks)
```

📌 **خروجی این کد:**  
✅ **پردازش‌هایی که اولویت بالاتری دارند، زودتر اجرا می‌شوند.**  
✅ **پردازش‌هایی که منابع کمتری مصرف می‌کنند، اولویت بیشتری می‌گیرند.**  

---

## **📌 آیا ماژول `scheduler` نیاز به بهینه‌سازی دارد؟**
📌 **پیشنهادات برای بهینه‌سازی:**  
🔹 **اضافه کردن قابلیت مدیریت صف پردازش‌های بلادرنگ** (Real-time Scheduling)  
🔹 **اضافه کردن پشتیبانی از الگوریتم‌های زمان‌بندی پیشرفته مانند `Round Robin` و `Fair Queueing`**  
🔹 **امکان تنظیم اولویت پردازش‌ها بر اساس میزان مصرف منابع و نیازمندی‌های آنی سیستم**  

---

### **🎯 نتیجه‌گیری**
✅ **ماژول `scheduler/` وظیفه مدیریت اولویت پردازش‌ها را دارد.**  
✅ **با کمک `PriorityManager` می‌توان پردازش‌های حیاتی را زودتر اجرا کرد و از تأخیر جلوگیری کرد.**  
✅ **این ماژول برای افزایش بهره‌وری پردازش داده‌ها در کل سیستم استفاده می‌شود.**  
✅ **با بهینه‌سازی این ماژول، می‌توان عملکرد زمان‌بندی را در مقیاس‌های بزرگ‌تر بهبود بخشید.**  

---