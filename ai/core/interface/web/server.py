from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn
import logging
from typing import Dict, Any

from ...learning.core_learner import CoreLearner
from ..api.internal_api import InternalAPI

logger = logging.getLogger(__name__)


class WebServer:
    def __init__(self, config: Dict[str, Any]):
        self.app = FastAPI()
        self.config = config
        self.base_path = Path(__file__).parent

        # تنظیم مسیرهای استاتیک و تمپلیت‌ها
        self.app.mount("/static", StaticFiles(directory=self.base_path / "static"), name="static")
        self.templates = Jinja2Templates(directory=self.base_path / "templates")

        # راه‌اندازی API داخلی
        self.api = InternalAPI(config)

        # تنظیم مسیرها
        self._setup_routes()

    def _setup_routes(self):
        @self.app.get("/")
        async def home(request: Request):
            return self.templates.TemplateResponse(
                "index.html",
                {"request": request}
            )

        # اتصال مسیرهای API داخلی
        self.app.mount("/api", self.api.app)

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        uvicorn.run(self.app, host=host, port=port)


def create_server(config: Dict[str, Any] = None) -> WebServer:
    if config is None:
        config = {}
    return WebServer(config)


if __name__ == "__main__":
    server = create_server()
    server.run()