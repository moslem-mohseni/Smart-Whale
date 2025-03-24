"""
ماژول semantic_metrics.py

این ماژول مسئول جمع‌آوری، تجمیع و گزارش متریک‌های مربوط به تحلیل معنایی متون فارسی است.
کلاس SemanticMetrics با استفاده از اجزای سیستم مانیتورینگ (Collector، Aggregator، Exporter، HealthChecker، HealthReporter، DashboardGenerator، AlertVisualizer)
متریک‌های مربوط به درخواست‌ها، تعداد تحلیل‌های معنایی، زمان پردازش، استفاده از مدل‌های هوشمند (smart model و teacher) و سایر آمارها را ثبت و گزارش می‌کند.
"""

import logging
import time
import json
from typing import Dict, Any, Optional

# واردات اجزای سیستم مانیتورینگ
from ai.core.monitoring.metrics.collector import MetricsCollector
from ai.core.monitoring.metrics.aggregator import MetricsAggregator
from ai.core.monitoring.metrics.exporter import MetricsExporter
from ai.core.monitoring.health.checker import HealthChecker
from ai.core.monitoring.health.reporter import HealthReporter
from ai.core.monitoring.visualization.dashboard_generator import DashboardGenerator
from ai.core.monitoring.visualization.alert_visualizer import AlertVisualizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SemanticMetrics:
    def __init__(self,
                 collector: Optional[MetricsCollector] = None,
                 aggregator: Optional[MetricsAggregator] = None,
                 exporter: Optional[MetricsExporter] = None,
                 health_checker: Optional[HealthChecker] = None,
                 health_reporter: Optional[HealthReporter] = None,
                 dashboard_generator: Optional[DashboardGenerator] = None,
                 alert_visualizer: Optional[AlertVisualizer] = None):
        self.logger = logger
        self.collector = collector
        self.aggregator = aggregator
        self.exporter = exporter
        self.health_checker = health_checker
        self.health_reporter = health_reporter
        self.dashboard_generator = dashboard_generator
        self.alert_visualizer = alert_visualizer

        # دیکشنری داخلی برای نگهداری متریک‌های تحلیل معنایی
        self.internal_metrics = {
            "requests": 0,
            "cache_hits": 0,
            "semantic_analyses": 0,
            "total_processing_time": 0.0,
            "smart_model_uses": 0,
            "teacher_uses": 0
        }
        self.logger.info("SemanticMetrics initialized.")

    def collect_semantic_metrics(self, text_length: int, processing_time: float, source: str) -> None:
        self.internal_metrics["requests"] += 1
        self.internal_metrics["semantic_analyses"] += 1
        self.internal_metrics["total_processing_time"] += processing_time
        if source == "smart_model":
            self.internal_metrics["smart_model_uses"] += 1
        elif source == "teacher":
            self.internal_metrics["teacher_uses"] += 1

        if self.collector:
            metric_data = {
                "semantic_analysis": {
                    "text_length": text_length,
                    "processing_time": processing_time,
                    "source": source,
                    "timestamp": time.time()
                }
            }
            try:
                self.collector.collect_metrics()
            except Exception as e:
                self.logger.error(f"Error collecting semantic metrics: {e}")

    def aggregate_and_export(self) -> None:
        if not self.aggregator:
            self.logger.warning("Aggregator not configured for SemanticMetrics.")
            return
        if not self.exporter:
            self.logger.warning("Exporter not configured for SemanticMetrics.")
            return
        try:
            self.aggregator.aggregate_metrics()
        except Exception as e:
            self.logger.error(f"Error aggregating semantic metrics: {e}")
            return
        try:
            self.exporter.export_metrics()
            self.logger.info("Semantic metrics aggregated and exported successfully.")
        except Exception as e:
            self.logger.error(f"Error exporting semantic metrics: {e}")

    def generate_dashboard(self) -> None:
        if not self.dashboard_generator:
            self.logger.warning("DashboardGenerator not configured for SemanticMetrics.")
            return
        try:
            self.dashboard_generator.generate_dashboard()
            self.logger.info("Semantic metrics dashboard generated successfully.")
        except Exception as e:
            self.logger.error(f"Error generating semantic metrics dashboard: {e}")

    def visualize_alerts(self) -> None:
        if not self.alert_visualizer:
            self.logger.warning("AlertVisualizer not configured for SemanticMetrics.")
            return
        try:
            self.alert_visualizer.visualize_alerts()
            self.logger.info("Semantic metrics alerts visualized successfully.")
        except Exception as e:
            self.logger.error(f"Error visualizing semantic alerts: {e}")

    def check_system_health(self) -> Dict[str, Any]:
        if not self.health_checker or not self.health_reporter:
            self.logger.warning("HealthChecker or HealthReporter not configured for SemanticMetrics.")
            return {"status": "unavailable", "details": "No health checker or reporter configured."}
        try:
            self.health_checker.run_health_checks()
            self.health_reporter.report_health()
            return {"status": "ok", "details": "Semantic metrics health checks performed and reported."}
        except Exception as e:
            self.logger.error(f"Error checking semantic metrics system health: {e}")
            return {"status": "error", "details": str(e)}

    def get_metrics_snapshot(self) -> Dict[str, Any]:
        return dict(self.internal_metrics)

    def reset_metrics(self) -> None:
        self.internal_metrics = {
            "requests": 0,
            "cache_hits": 0,
            "semantic_analyses": 0,
            "total_processing_time": 0.0,
            "smart_model_uses": 0,
            "teacher_uses": 0
        }
        self.logger.info("SemanticMetrics internal metrics reset.")


if __name__ == "__main__":
    # تست ساده برای بررسی عملکرد
    sm = SemanticMetrics()
    sm.collect_semantic_metrics(text_length=200, processing_time=0.5, source="smart_model")
    snapshot = sm.get_metrics_snapshot()
    print("Semantic Metrics Snapshot:", json.dumps(snapshot, ensure_ascii=False, indent=2))
