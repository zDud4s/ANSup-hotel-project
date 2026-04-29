"""ANSup hotel-booking clustering project.

The ``_windows_compat`` import below MUST run before any submodule imports
pandas; it patches ``platform.machine()`` to bypass a Python 3.13 WMI hang
on Windows 11 24H2/25H2 (where Microsoft removed ``wmic.exe``).
"""

from . import _windows_compat as _windows_compat  # noqa: F401
