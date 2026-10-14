from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from translatorx import logging_setup


def test_fault_log_records_stack_and_survives_restarts(tmp_path, monkeypatch):
    monkeypatch.setenv("PYAPPIFY_APP_VERSION", "v-test-crash")
    path = tmp_path / "translatorx.log"
    script = """
import faulthandler
import sys
from pathlib import Path
from translatorx import logging_setup

logging_setup.log_path = lambda: Path(sys.argv[1])
logging_setup.setup_logging()
faulthandler.dump_traceback(file=logging_setup._fault_log)
"""
    for _launch in range(2):
        subprocess.run(
            [sys.executable, "-c", script, str(path)],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )

    fault_log = (tmp_path / "translatorx-crash.log").read_text(encoding="utf-8")
    assert fault_log.count("=== TranslatorX") == 2
    assert fault_log.count('File "<string>"') == 2
    assert "运行版本：v-test-crash" in path.read_text(encoding="utf-8")


def test_unwritable_fault_log_does_not_prevent_startup(tmp_path, caplog):
    path = tmp_path / "missing-directory" / "translatorx-crash.log"
    assert logging_setup._enable_fault_logging(path) is None
    assert "无法启用原生崩溃堆栈记录" in caplog.text
