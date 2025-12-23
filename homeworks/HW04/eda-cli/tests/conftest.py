"""Конфигурация pytest для src-структуры проекта.

Тесты импортируют пакет eda_cli (см. tests/test_core.py), поэтому добавляем
каталог src в sys.path для корректного импорта при запуске из IDE/CLI. [file:19]
"""
from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_sys_path() -> None:
    """Гарантирует, что каталог src присутствует в sys.path."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_ensure_src_on_sys_path()
