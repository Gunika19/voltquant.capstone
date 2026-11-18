"""VoltQuant Hardware Communication Link (hardware/vq_send.py)
Re-exports VoltQuantLink for modular packaging.
"""

from pathlib import Path
import sys

# Ensure root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vq_send import VoltQuantLink, HAS_PYSERIAL

__all__ = ["VoltQuantLink", "HAS_PYSERIAL"]
