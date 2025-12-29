"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# for register purpose
from . import optim
try:
    from . import data
except Exception:
    # Allow inference-only usage without dataset dependencies (e.g., pycocotools on Windows).
    data = None
from . import nn
from . import zoo
