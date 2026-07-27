"""Start the local API and dashboard without attaching them to the caller."""

from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
DATA_DIR = ROOT / "data"
SERVICES = (
    (
        "API",
        8000,
        ("-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000"),
        "api.log",
        "api-error.log",
    ),
    (
        "Streamlit",
        8501,
        (
            "-m",
            "streamlit",
            "run",
            "dashboard.py",
            "--server.address",
            "127.0.0.1",
            "--server.port",
            "8501",
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ),
        "streamlit.log",
        "streamlit-error.log",
    ),
)


def _port_is_open(port: int) -> bool:
    with socket.socket() as connection:
        connection.settimeout(0.2)
        return connection.connect_ex(("127.0.0.1", port)) == 0


def _start_service(
    name: str,
    port: int,
    arguments: tuple[str, ...],
    stdout_name: str,
    stderr_name: str,
) -> None:
    if _port_is_open(port):
        print(f"{name}: already running on http://127.0.0.1:{port}")
        return

    flags = (
        subprocess.CREATE_NEW_PROCESS_GROUP
        | subprocess.DETACHED_PROCESS
        | subprocess.CREATE_NO_WINDOW
    )
    with (
        (DATA_DIR / stdout_name).open("ab") as stdout,
        (DATA_DIR / stderr_name).open("ab") as stderr,
    ):
        process = subprocess.Popen(
            [str(PYTHON), *arguments],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
            creationflags=flags,
        )
    print(f"{name}: started PID {process.pid} on http://127.0.0.1:{port}")


def main() -> int:
    if not PYTHON.exists():
        print(f"Virtual environment Python not found: {PYTHON}", file=sys.stderr)
        return 1
    DATA_DIR.mkdir(exist_ok=True)
    for service in SERVICES:
        _start_service(*service)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
