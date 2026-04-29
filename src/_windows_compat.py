"""Windows 11 platform.machine() compatibility shim.

Python 3.13's ``platform.machine()`` resolves the CPU architecture by
querying WMI via ``wmic.exe``. Microsoft removed ``wmic.exe`` in Windows 11
24H2/25H2, so on those releases the call hangs for minutes (or until killed)
on every ``import pandas`` (pandas reaches it via
``pandas/compat/_constants.py: WASM = ... platform.machine() in [...]``).

This module replaces ``platform.machine`` with a constant taken from the
``PROCESSOR_ARCHITECTURE`` environment variable (which Windows always sets),
bypassing the WMI chain.

It also sets ``LOKY_MAX_CPU_COUNT`` when missing. scikit-learn reaches
joblib/loky during k-means fitting, and loky otherwise launches the same WMI
probe to count physical CPU cores.

Imported by ``src/__init__.py`` so the patch is applied before any submodule
imports pandas.
"""

from __future__ import annotations

import os
import platform
import sys


def _apply_patch() -> None:
    if sys.platform != "win32":
        return
    arch = os.environ.get("PROCESSOR_ARCHITECTURE") or "AMD64"
    platform.machine = lambda: arch
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")


_apply_patch()
