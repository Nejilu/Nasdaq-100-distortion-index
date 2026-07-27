import os
import subprocess
import sys
from pathlib import Path

import run_local


def test_launcher_reuses_the_active_python_interpreter():
    assert run_local.PYTHON == Path(sys.executable)


def test_posix_launcher_starts_a_detached_session():
    assert run_local._process_creation_options("posix") == {
        "creationflags": 0,
        "start_new_session": True,
    }


def test_windows_launcher_uses_detached_process_flags():
    if os.name != "nt":
        return

    options = run_local._process_creation_options("nt")

    assert options == {
        "creationflags": (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NO_WINDOW
        )
    }
