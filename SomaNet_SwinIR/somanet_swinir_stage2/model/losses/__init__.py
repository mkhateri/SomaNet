from .losses import (
    BinaryFocalLossWithLogits,
    BCE_loss_func,
    BCEWithLogitsLoss,
    MAELoss,
    MSELoss,
    WeightedMSE,
    embedding_single_offset_loss,
    embedding_loss,
    embedding_loss_teacher_student,
    CrossEntropyLoss,
    DiceLoss,
)

# Register available losses
LOSS_REGISTRY = {
    "BinaryFocalLossWithLogits": BinaryFocalLossWithLogits,
    "BCE_loss_func": BCE_loss_func,
    "BCEWithLogitsLoss": BCEWithLogitsLoss,
    "MAELoss": MAELoss,
    "MSELoss": MSELoss,
    "WeightedMSE": WeightedMSE,
    "embedding_single_offset_loss": embedding_single_offset_loss,
    "embedding_loss": embedding_loss,
    "embedding_loss_teacher_student": embedding_loss_teacher_student,
    "CrossEntropyLoss": CrossEntropyLoss,
    "DiceLoss": DiceLoss,
}

__all__ = [
    "BinaryFocalLossWithLogits",
    "BCE_loss_func",
    "BCEWithLogitsLoss",
    "MAELoss",
    "MSELoss",
    "WeightedMSE",
    "embedding_single_offset_loss",
    "embedding_loss",
    "embedding_loss_teacher_student",
    "CrossEntropyLoss",
    "DiceLoss",
    "LOSS_REGISTRY",
]
