# infrastructure/clickhouse/adapters/load_balancer.py
import random
import logging
import time
from typing import Dict, List, Optional, Any
from collections import deque
from clickhouse_driver import Client
from clickhouse_driver.errors import Error as ClickHouseDriverError
from ..config.config import config
from ..exceptions import ConnectionError, AuthenticationError, OperationalError

logger = logging.getLogger(__name__)


class ClickHouseLoadBalancer:
    """
    مدیریت Load Balancing برای ClickHouse با پشتیبانی از چندین استراتژی

    این کلاس امکان توزیع بار بین چندین سرور ClickHouse را با استراتژی‌های مختلف
    (تصادفی، دوره‌ای، اتصال کمتر) فراهم می‌کند.
    """

    def __init__(self, custom_config=None):
        """
        مقداردهی اولیه Load Balancer با استفاده از تنظیمات متمرکز

        Args:
            custom_config (ClickHouseConfig, optional): تنظیمات سفارشی اتصال به ClickHouse

        Raises:
            OperationalError: در صورت بروز خطا در مقداردهی اولیه
        """
        try:
            self.config = custom_config or config
            self.servers = self.config.get_servers()
            self.mode = self.config.load_balancer_mode
            self.connection_params = self.config.get_connection_params()

            # حذف host از پارامترهای اتصال، چون باید برای هر سرور جداگانه تنظیم شود
            connection_params_without_host = self.connection_params.copy()
            if 'host' in connection_params_without_host:
                del connection_params_without_host['host']
            self.connection_params_without_host = connection_params_without_host

            # ساختارهای داده برای پشتیبانی از استراتژی‌های مختلف
            self.connections: Dict[str, Client] = {}  # ذخیره اتصالات فعال
            self.server_failures: Dict[str, int] = {server: 0 for server in self.servers}  # تعداد خطاهای هر سرور
            self.server_queue = deque(self.servers)  # صف سرورها برای round-robin
            self.active_connections: Dict[str, int] = {server: 0 for server in
                                                       self.servers}  # تعداد اتصالات فعال هر سرور

            logger.info(f"ClickHouse Load Balancer initialized with mode={self.mode}, servers={self.servers}")

        except Exception as e:
            error_msg = f"Failed to initialize LoadBalancer: {str(e)}"
            logger.error(error_msg)
            raise OperationalError(
                message=error_msg,
                code="LB001"
            )

    def _get_random_server(self) -> str:
        """
        انتخاب تصادفی یک سرور

        Returns:
            str: آدرس سرور انتخاب شده
        """
        return random.choice(self.servers)

    def _get_round_robin_server(self) -> str:
        """
        انتخاب سرور با الگوریتم round-robin

        Returns:
            str: آدرس سرور انتخاب شده
        """
        server = self.server_queue.popleft()
        self.server_queue.append(server)
        return server

    def _get_least_conn_server(self) -> str:
        """
        انتخاب سرور با کمترین تعداد اتصال فعال

        Returns:
            str: آدرس سرور انتخاب شده
        """
        return min(self.active_connections.items(), key=lambda x: x[1])[0]

    def _create_connection(self, server: str) -> Client:
        """
        ایجاد یک اتصال جدید به سرور مشخص شده

        Args:
            server (str): آدرس سرور

        Returns:
            Client: شیء اتصال به ClickHouse

        Raises:
            ConnectionError: در صورت بروز خطا در برقراری اتصال
            AuthenticationError: در صورت بروز خطا در احراز هویت
        """
        try:
            conn = Client(host=server, **self.connection_params_without_host)
            # تست اتصال با یک کوئری ساده
            conn.execute("SELECT 1")
            return conn
        except ClickHouseDriverError as e:
            self.server_failures[server] += 1
            error_str = str(e).lower()

            # تشخیص نوع خطا
            if "authentication" in error_str or "password" in error_str or "access" in error_str:
                raise AuthenticationError(
                    message=f"Authentication failed for server {server}: {str(e)}",
                    code="LB002",
                    host=server,
                    user=self.connection_params_without_host.get('user', 'unknown')
                )
            else:
                raise ConnectionError(
                    message=f"Failed to connect to server {server}: {str(e)}",
                    code="LB003",
                    host=server
                )

    def get_connection(self) -> Client:
        """
        دریافت یک اتصال از سرور بهینه با توجه به استراتژی load balancing

        Returns:
            Client: شیء اتصال به ClickHouse

        Raises:
            ConnectionError: در صورت عدم موفقیت در برقراری اتصال به هیچ سرور
            AuthenticationError: در صورت بروز خطا در احراز هویت
            OperationalError: در صورت بروز سایر خطاها
        """
        try:
            # انتخاب سرور با توجه به استراتژی load balancing
            if self.mode == "round-robin":
                server = self._get_round_robin_server()
            elif self.mode == "least-conn":
                server = self._get_least_conn_server()
            else:  # حالت "random" یا هر مقدار دیگر
                server = self._get_random_server()

            # ایجاد اتصال اگر وجود نداشته باشد
            if server not in self.connections or self.connections[server] is None:
                self.connections[server] = self._create_connection(server)

            # افزایش شمارنده اتصالات فعال
            self.active_connections[server] += 1

            logger.debug(f"Connection obtained from server {server}")
            return self.connections[server]

        except (ConnectionError, AuthenticationError):
            # خطاهای اتصال و احراز هویت را مستقیماً منتقل می‌کنیم
            raise
        except Exception as e:
            error_msg = f"Failed to get connection from load balancer: {str(e)}"
            logger.error(error_msg)
            raise OperationalError(
                message=error_msg,
                code="LB004"
            )

    def release_connection(self, server: str):
        """
        آزادسازی یک اتصال برای سرور مشخص

        Args:
            server (str): نام سرور
        """
        if server in self.active_connections:
            self.active_connections[server] = max(0, self.active_connections[server] - 1)
            logger.debug(f"Connection released for server {server}")

    def close_all_connections(self):
        """
        بستن تمام اتصالات فعال

        Raises:
            OperationalError: در صورت بروز خطا در بستن اتصالات
        """
        try:
            for server, conn in self.connections.items():
                if conn:
                    try:
                        conn.disconnect()
                    except Exception as e:
                        logger.warning(f"Error closing connection to {server}: {str(e)}")

            self.connections.clear()
            self.active_connections = {server: 0 for server in self.servers}
            logger.info("All LoadBalancer connections closed")

        except Exception as e:
            error_msg = f"Error closing LoadBalancer connections: {str(e)}"
            logger.error(error_msg)
            raise OperationalError(
                message=error_msg,
                code="LB005"
            )

    def get_stats(self) -> Dict[str, Any]:
        """
        دریافت آمار استفاده از Load Balancer

        Returns:
            Dict[str, Any]: آمار استفاده از سرورها و اتصالات
        """
        stats = {
            "mode": self.mode,
            "servers": len(self.servers),
            "active_connections": self.active_connections,
            "server_failures": self.server_failures
        }
        return stats
