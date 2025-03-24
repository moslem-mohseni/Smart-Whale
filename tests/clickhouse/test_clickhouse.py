import pytest
from unittest.mock import MagicMock, patch
from infrastructure.clickhouse.config import ClickHouseConfig
from infrastructure.clickhouse.adapters import ClickHouseAdapter, ClickHouseConnectionPool
from infrastructure.clickhouse.service import AnalyticsService
from infrastructure.clickhouse.optimization import QueryOptimizer, CacheManager


# Mock کردن تمام وابستگی‌های پرومتئوس
@pytest.fixture(autouse=True)
def disable_prometheus():
    """ غیرفعال کردن پرومتئوس در تست‌ها """
    with patch("infrastructure.clickhouse.monitoring.PrometheusExporter", new=MagicMock()):
        yield


@pytest.fixture
def config():
    return ClickHouseConfig()


@pytest.fixture
def connection_pool(config):
    return ClickHouseConnectionPool(config)


@pytest.fixture
def adapter(config):
    return ClickHouseAdapter(config)


@pytest.fixture
def query_optimizer():
    return QueryOptimizer()


@pytest.fixture
def cache_manager():
    return CacheManager()


@pytest.fixture
def analytics_service(adapter, cache_manager, query_optimizer):
    return AnalyticsService(adapter, cache_manager, query_optimizer)


def test_clickhouse_connection(adapter):
    """ تست اتصال به ClickHouse """
    connection = adapter.connection_pool.get_connection()
    assert connection is not None
    adapter.connection_pool.release_connection(connection)


def test_execute_query(adapter):
    """ تست اجرای یک کوئری ساده در ClickHouse """
    query = "SELECT 1"
    result = adapter.execute(query)
    assert result is not None


def test_analytics_service(analytics_service):
    """ تست اجرای کوئری تحلیلی بدون وابستگی به Prometheus """
    mock_query = "SELECT COUNT(*) FROM system.tables"
    analytics_service.clickhouse_adapter.execute = MagicMock(return_value=[{"count": 10}])

    result = analytics_service.execute_analytics_query(mock_query)
    assert result == [{"count": 10}]
