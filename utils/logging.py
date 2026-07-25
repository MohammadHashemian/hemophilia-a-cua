import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from utils.path_utils import get_project_root


class PrettyFormatter(logging.Formatter):
    def format(self, record):
        ts = self.formatTime(record, "%H:%M:%S")
        base = f"[{ts}] {record.levelname:<8} {record.getMessage()}"

        context = []
        for field in ("run_id", "scenario", "worker_id"):
            value = getattr(record, field, None)
            if value is not None:
                context.append(f"{field}={value}")

        if context:
            base += " | " + ", ".join(context)

        return base


def setup_root_logger(
    level=logging.INFO,
    log_file: Path = get_project_root() / "logs/simulation.log",
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
) -> logging.Logger:

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[],
        force=True,
    )

    root = logging.getLogger()
    root.handlers.clear()

    formatter = PrettyFormatter()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_file,
        mode="a",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root.addHandler(console)
    root.addHandler(file_handler)

    logging.getLogger("urllib3").setLevel(level)
    logging.getLogger("asyncio").setLevel(level)

    return root
