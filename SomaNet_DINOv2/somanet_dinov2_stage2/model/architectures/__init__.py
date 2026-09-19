# model/architectures/__init__.py — DINOv2Seg is the only architecture used here.
from .models import DINOv2Seg

ARCHITECTURE_REGISTRY = {"DINOv2Seg": DINOv2Seg}

__all__ = ["DINOv2Seg", "ARCHITECTURE_REGISTRY"]
