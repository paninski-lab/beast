"""Fixtures for beast/preprocess/segment tests.

Mocks `accelerate` in sys.modules before any test module imports it, so that
beast.preprocess.segment.sam3 (which does ``from accelerate import Accelerator``
at module level) can be imported in environments where accelerate is not installed.
"""

import sys
from unittest.mock import MagicMock

sys.modules.setdefault('accelerate', MagicMock())
