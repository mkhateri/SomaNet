# model/architectures/__init__.py

from .models import (
    #ResidualUNet2D_deep,
    #ResidualUNet2D_affs2,
    #ResidualUNet2D_embedding,
    #Unet2d,
    SwinIR,
    # Add other models here as needed
)

# Register available architectures
ARCHITECTURE_REGISTRY = {
    # "ResidualUNet2D_deep": ResidualUNet2D_deep,
    # "ResidualUNet2D_affs2": ResidualUNet2D_affs2,
    # "ResidualUNet2D_embedding": ResidualUNet2D_embedding,
    # "Unet2d": Unet2d,
    "SwinIR": SwinIR,
    # Add any additional architectures to the registry
}

__all__ = [
    # "ResidualUNet2D_deep",
    # "ResidualUNet2D_affs2",
    # "ResidualUNet2D_embedding",
    # "Unet2d",
    "SwinIR",
    "ARCHITECTURE_REGISTRY",
]



