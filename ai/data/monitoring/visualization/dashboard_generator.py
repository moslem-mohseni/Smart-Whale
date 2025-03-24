# dashboard_generator.py
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import prometheus_client
import requests

# راه‌اندازی اپلیکیشن داشبورد
app = dash.Dash(__name__)

# آدرس سرویس Prometheus که متریک‌ها را ذخیره می‌کند
PROMETHEUS_URL = "http://localhost:8000/metrics"

def fetch_metrics():
    """
    دریافت داده‌های متریک‌ها از Prometheus Exporter.
    """
    try:
        response = requests.get(PROMETHEUS_URL)
        response.raise_for_status()
        raw_data = response.text
        metrics = parse_metrics(raw_data)
        return metrics
    except requests.exceptions.RequestException as e:
        print(f"❌ خطا در دریافت متریک‌ها: {e}")
        return {}

def parse_metrics(raw_data):
    """
    پردازش داده‌های متریک از فرمت متنی Prometheus به دیکشنری.
    """
    metrics = {}
    for line in raw_data.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) == 2:
            try:
                metrics[parts[0]] = float(parts[1])
            except ValueError:
                continue
    return metrics

# طراحی UI داشبورد
app.layout = html.Div(children=[
    html.H1(children="📊 داشبورد مانیتورینگ Smart Whale", style={'textAlign': 'center'}),

    # نمایش متریک‌های سیستمی
    html.Div([
        html.H3("🔍 استفاده از CPU"),
        dcc.Graph(id="cpu-usage-graph"),
    ]),

    html.Div([
        html.H3("📊 استفاده از حافظه"),
        dcc.Graph(id="memory-usage-graph"),
    ]),

    html.Div([
        html.H3("💾 عملیات I/O دیسک"),
        dcc.Graph(id="disk-io-graph"),
    ]),

    html.Div([
        html.H3("🌐 ترافیک شبکه"),
        dcc.Graph(id="network-io-graph"),
    ]),

    # رفرش خودکار داده‌ها
    dcc.Interval(id="interval-component", interval=5000, n_intervals=0)
])

@app.callback(
    [
        Output("cpu-usage-graph", "figure"),
        Output("memory-usage-graph", "figure"),
        Output("disk-io-graph", "figure"),
        Output("network-io-graph", "figure")
    ],
    Input("interval-component", "n_intervals")
)
def update_dashboard(n):
    """
    به‌روزرسانی نمودارهای داشبورد با داده‌های جدید.
    """
    metrics = fetch_metrics()

    figures = []
    for metric_name in ["cpu_usage", "memory_usage", "disk_io", "network_io"]:
        values = [metrics.get(metric_name, 0)]
        figure = {
            "data": [{"x": ["لحظه‌ای"], "y": values, "type": "bar", "name": metric_name}],
            "layout": {"title": f"{metric_name} (%)"}
        }
        figures.append(figure)

    return figures

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
