# report_generator.py
import json
import csv
from fpdf import FPDF
from typing import Dict, List
from datetime import datetime
import os

class ReportGenerator:
    """
    کلاس برای تولید گزارش‌های متریک‌ها و هشدارها در قالب JSON، CSV و PDF.
    """

    def __init__(self, output_dir: str = "reports/"):
        """
        مقداردهی اولیه کلاس.

        :param output_dir: دایرکتوری ذخیره گزارش‌ها.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_json_report(self, data: Dict[str, float], filename: str = None) -> str:
        """
        تولید گزارش متریک‌ها در قالب JSON.

        :param data: دیکشنری شامل متریک‌ها.
        :param filename: نام فایل (در صورت عدم ورود، تاریخ فعلی انتخاب می‌شود).
        :return: مسیر ذخیره فایل JSON.
        """
        filename = filename or f"metrics_report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

        print(f"✅ گزارش JSON ذخیره شد: {filepath}")
        return filepath

    def generate_csv_report(self, data: Dict[str, float], filename: str = None) -> str:
        """
        تولید گزارش متریک‌ها در قالب CSV.

        :param data: دیکشنری شامل متریک‌ها.
        :param filename: نام فایل (در صورت عدم ورود، تاریخ فعلی انتخاب می‌شود).
        :return: مسیر ذخیره فایل CSV.
        """
        filename = filename or f"metrics_report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Metric", "Value"])
            for key, value in data.items():
                writer.writerow([key, value])

        print(f"✅ گزارش CSV ذخیره شد: {filepath}")
        return filepath

    def generate_pdf_report(self, data: Dict[str, float], filename: str = None) -> str:
        """
        تولید گزارش متریک‌ها در قالب PDF.

        :param data: دیکشنری شامل متریک‌ها.
        :param filename: نام فایل (در صورت عدم ورود، تاریخ فعلی انتخاب می‌شود).
        :return: مسیر ذخیره فایل PDF.
        """
        filename = filename or f"metrics_report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf"
        filepath = os.path.join(self.output_dir, filename)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, "گزارش متریک‌های سیستم", ln=True, align="C")

        pdf.set_font("Arial", "", 12)
        pdf.ln(10)
        for key, value in data.items():
            pdf.cell(200, 10, f"{key}: {value}%", ln=True, align="L")

        pdf.output(filepath)
        print(f"✅ گزارش PDF ذخیره شد: {filepath}")
        return filepath

if __name__ == "__main__":
    generator = ReportGenerator()

    # داده‌های نمونه برای گزارش
    sample_data = {
        "cpu_usage": 75.4,
        "memory_usage": 68.2,
        "disk_io": 120.5,
        "network_io": 95.3
    }

    generator.generate_json_report(sample_data)
    generator.generate_csv_report(sample_data)
    generator.generate_pdf_report(sample_data)
