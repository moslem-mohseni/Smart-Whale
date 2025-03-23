import pytest
from infrastructure.clickhouse.config import ClickHouseConfig
from infrastructure.clickhouse.adapters import ClickHouseAdapter, ClickHouseConnectionPool
from infrastructure.clickhouse.service import AnalyticsService
from infrastructure.clickhouse.optimization import QueryOptimizer, CacheManager
from infrastructure.clickhouse.security import AccessControl
from infrastructure.clickhouse.monitoring import HealthCheck

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
def analytics_service(adapter):
    return AnalyticsService(adapter)

@pytest.fixture
def cache_manager():
    return CacheManager()

@pytest.fixture
def query_optimizer():
    return QueryOptimizer()

@pytest.fixture
def access_control():
    return AccessControl()

@pytest.fixture
def health_check():
    return HealthCheck()

def test_connection_pool(connection_pool):
    connection = connection_pool.get_connection()
    assert connection is not None
    connection_pool.release_connection(connection)

def test_query_execution(adapter):
    query = "SELECT 1"
    result = adapter.execute(query)
    assert result is not None

def test_cache_functionality(cache_manager):
    query = "SELECT 1"
    result = {"data": [1]}
    cache_manager.set_cached_result(query, result, ttl=60)
    cached_result = cache_manager.get_cached_result(query)
    assert cached_result == result

def test_query_optimization(query_optimizer):
    raw_query = "SELECT * FROM users"
    optimized_query = query_optimizer.optimize_query(raw_query)
    assert "SELECT *" not in optimized_query

def test_access_control(access_control):
    token = access_control.generate_token("test_user", "admin")
    decoded = access_control.verify_token(token)
    assert decoded is not None
    assert decoded["username"] == "test_user"

def test_health_check(health_check):
    status = health_check.check_system_health()
    assert status["status"] in ["healthy", "unhealthy"]
