import networkx as nx

class DependencyManager:
    """
    ماژولی برای مدیریت وابستگی‌های بین پردازش‌های داده‌ای.
    این ماژول ترتیب بهینه اجرای پردازش‌ها را با استفاده از تحلیل گراف وابستگی مشخص می‌کند.
    """

    def __init__(self):
        self.dependency_graph = nx.DiGraph()  # گراف جهت‌دار برای مدل‌سازی وابستگی پردازش‌ها

    def add_dependency(self, task_id: str, depends_on: list):
        """
        اضافه کردن یک پردازش و وابستگی‌های آن به گراف.

        :param task_id: شناسه پردازش
        :param depends_on: لیستی از شناسه پردازش‌هایی که این پردازش به آن‌ها وابسته است.
        """
        self.dependency_graph.add_node(task_id)
        for dependency in depends_on:
            self.dependency_graph.add_edge(dependency, task_id)

    def get_execution_order(self) -> list:
        """
        تعیین ترتیب بهینه اجرای پردازش‌ها با در نظر گرفتن وابستگی‌ها.

        :return: لیستی از پردازش‌ها به ترتیب اجرای بهینه.
        """
        try:
            execution_order = list(nx.topological_sort(self.dependency_graph))
            return execution_order
        except nx.NetworkXUnfeasible:
            return {"error": "Circular dependency detected. Execution order cannot be determined."}

    def has_circular_dependency(self) -> bool:
        """
        بررسی اینکه آیا در گراف وابستگی حلقه (Circular Dependency) وجود دارد.

        :return: مقدار True اگر حلقه وجود داشته باشد، در غیر این صورت False.
        """
        return not nx.is_directed_acyclic_graph(self.dependency_graph)

    def resolve_circular_dependency(self):
        """
        حذف وابستگی‌های حلقوی از گراف برای جلوگیری از مشکلات اجرا.
        """
        while not nx.is_directed_acyclic_graph(self.dependency_graph):
            cycle = next(nx.simple_cycles(self.dependency_graph), None)
            if cycle:
                self.dependency_graph.remove_edge(cycle[0], cycle[1])  # حذف یک یال برای شکستن حلقه

