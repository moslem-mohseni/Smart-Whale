# __init__.py
from .dashboard_generator import app as DashboardApp
from .report_generator import ReportGenerator
from .trend_visualizer import TrendVisualizer

__all__ = ["DashboardApp", "ReportGenerator", "TrendVisualizer"]
