import atexit
import json
import os
from datetime import datetime
from pathlib import Path
from threading import Lock


class PerformanceLogger:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
                atexit.register(cls._instance.save_to_file)
                cls._instance.metrics_log = []
        return cls._instance

    @staticmethod
    def find_project_root(marker_files=("pyproject.toml", "setup.py")):
        current_dir = Path(__file__).resolve().parent
        while current_dir != current_dir.root:
            if any((current_dir / marker).exists() for marker in marker_files):
                return current_dir
            current_dir = current_dir.parent
        return None

    def log(self, metric_name: str, function_name: str, value: str, custom_msg=None):
        self.metrics_log.append(
            {
                "metric": metric_name,
                "function_name": function_name,
                "value": value,
                "custom_msg": custom_msg,
            }
        )

    def save_to_file(self):
        if not self.metrics_log:
            return

        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".json"
        project_root = self.find_project_root()

        if project_root:
            folder_path = project_root / "performance_logs"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            filepath = folder_path / filename
            with open(filepath, "w") as f:
                json.dump(self.metrics_log, f, indent=4, ensure_ascii=False)
            print(f"Performance log saved to {filename}")
