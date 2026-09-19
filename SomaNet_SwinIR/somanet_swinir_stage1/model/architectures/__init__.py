# model/architectures/__init__.py
# SwinIR-only architecture registry.

from .models import SwinIR

# Register available architectures
ARCHITECTURE_REGISTRY = {
    "SwinIR": SwinIR,
}

__all__ = [
    "SwinIR",
    "ARCHITECTURE_REGISTRY",
]
