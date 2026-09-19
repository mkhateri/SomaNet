import torch
import numpy as np
import torch.nn as nn
import monai.losses as monai_losses
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Variable


class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Initialize the BinaryFocalLossWithLogits class.

        Parameters:
        - alpha (float): Weighting factor for the positive class.
        - gamma (float): Focusing parameter to reduce the relative loss for well-classified examples.
        - reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(BinaryFocalLossWithLogits, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass for the loss computation.

        Parameters:
        - input (Tensor): Predicted logits.
        - target (Tensor): Ground truth labels.

        Returns:
        - Tensor: Computed focal loss.
        """
        # Compute the binary cross-entropy loss with logits
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(input)
        
        # Compute p_t, the probability assigned to the true label
        p_t = p * target + (1 - p) * (1 - target)
        
        # Compute the focal loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        
        # Apply reduction method
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def BCE_loss_func(output,target, weight_rate=[1,1]):
    # print("weight_rate",weight_rate)
    weight = torch.FloatTensor([torch.sum(target == 1).item(), torch.sum(target == 0).item()]).cuda()

    loss_fn = nn.CrossEntropyLoss(weight=weight)
    # loss_fn = nn.BCELoss(weight=weight)
    loss = loss_fn(output, target.squeeze(1).long())
    return loss

# BCEWithLogitsLoss sigmoid = False; it internally considered sigmoid 
class BCEWithLogitsLoss(nn.Module):
    def __init__(self, sigmoid=False, auto_pos_weight=True):
        super(BCEWithLogitsLoss, self).__init__()
        self.sigmoid = sigmoid
        self.auto_pos_weight = auto_pos_weight

    def forward(self, preds, targets):
        # Automatically calculate pos_weight if enabled
        if self.auto_pos_weight:
            pos_weight = torch.sum(targets == 0).float() / torch.sum(targets == 1).float()
            pos_weight = pos_weight.clone().detach().to(targets.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        if self.sigmoid:
            preds = torch.sigmoid(preds)
        
        return criterion(preds, targets)


# L1 Loss
class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, x_gt, x_pred):
        return self.criterion(x_gt, x_pred)

# L2 Loss
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, x_gt, x_pred):
        return self.criterion(x_gt, x_pred)


# Weighted L2 Loss
class WeightedMSE(nn.Module):
    def __init__(self, eps=1e-8):
        super(WeightedMSE, self).__init__()
        self.eps = eps

    def forward(self, pred, target, weight=None):
        # Use float32 for stable computation
        pred_f32 = pred.float()
        target_f32 = target.float()

        s1 = torch.prod(torch.tensor(pred_f32.size()[2:]).float())
        s2 = pred_f32.size()[0]
        norm_term = (s1 * s2).to(pred_f32.device) + self.eps  # Add eps to avoid division by zero

        if weight is None:
            loss = torch.sum((pred_f32 - target_f32) ** 2) / norm_term
        else:
            weight_f32 = weight.float()
            loss = torch.sum(weight_f32 * (pred_f32 - target_f32) ** 2) / norm_term

        # Clamp loss to prevent extreme values
        loss = torch.clamp(loss, max=1e6)

        return loss


# ------------------------------------------------------------------------------------------
# Copied from https://github.com/weih527/Pixel-Embedded-Affinity, and adapted to our codebase
# Learning to Model Pixel-Embedded Affinity for Homogeneous Instance Segmentation (DOI: https://doi.org/10.1609/aaai.v36i1.19984)

def embedding_single_offset_loss(embedding, offset, target, weightmap=None, mask=None, criterion='WeightedMSE', mode='AAAI'):
    embedding_shift = torch.roll(embedding, shifts=tuple(offset), dims=(2, 3))
    if mode == 'AAAI':
        affs_temp = torch.sum(embedding_shift * embedding, dim=1)
    else:
        dis = nn.CosineSimilarity(dim=1, eps=1e-6)
        affs_temp = dis(embedding_shift, embedding)

    if criterion == 'WeightedMSE':
        criterion = WeightedMSE()
    elif criterion == 'MSELoss':
        criterion = MSELoss()
    elif criterion == 'MAELoss':
        criterion = MAELoss()
    else: 
        raise NotImplementedError(f"The criterion '{criterion}' is not implemented.")

    loss_temp = criterion(affs_temp*mask, target*mask, weightmap)

    return loss_temp, affs_temp

def embedding_loss(embedding, target, weightmap, mask, offsets, criterion='WeightedMSE', affs0_weight=1, mode='AAAI'):
    """
    Compute embedding/affinity loss with numerical stability improvements.
    Uses float32 internally for stable computation even when inputs are bfloat16.
    """
    # Convert to float32 for stable computation
    embedding_f32 = embedding.float()
    target_f32 = target.float()
    weightmap_f32 = weightmap.float() if weightmap is not None else None
    mask_f32 = mask.float()

    if mode == 'AAAI':
        embedding_f32 = F.normalize(embedding_f32, p=2, dim=1, eps=1e-8)

    affs = torch.zeros_like(target_f32)
    loss = torch.tensor(0.0, dtype=torch.float32, device=embedding.device)
    all_loss = []

    if affs0_weight == 1:
        affs0_weight_factor = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    elif affs0_weight == 2:
        affs0_weight_factor = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    elif affs0_weight == 3:
        affs0_weight_factor = [0.25, 0.25, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    else:
        affs0_weight_factor = [affs0_weight, affs0_weight, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    for i, offset in enumerate(offsets):
        shift_off = [-x for x in offset]
        loss_temp, affs_temp = embedding_single_offset_loss(
            embedding_f32, shift_off, target_f32[:, i],
            weightmap_f32[:, i] if weightmap_f32 is not None else None,
            mask_f32[:, i], criterion, mode=mode
        )

        # Check for NaN/Inf and skip if detected
        if torch.isnan(loss_temp) or torch.isinf(loss_temp):
            print(f"WARNING: NaN/Inf detected in embedding loss for offset {i}, skipping")
            all_loss.append(0.0)
            continue

        loss = loss + loss_temp
        all_loss.append(loss_temp.item())
        affs[:, i] = affs_temp

    return loss, affs, all_loss




def embedding_loss_teacher_student(embedding1, embedding2, mode='euclidean', weight=None, reduction='mean'):
    """
    Compute the loss between two embeddings based on the chosen similarity or distance metric.

    :param embedding1: First embedding tensor of shape (B, C, H, W).
    :param embedding2: Second embedding tensor of shape (B, C, H, W).
    :param mode: The similarity or distance metric to use. Options: 'cosine', 'dot', 'euclidean'.
    :param weight: Optional weight map of shape (B, H, W) to apply element-wise weight to the loss.
    :param reduction: Reduction method to apply to the loss. Options: 'mean', 'sum'.
    :return: The computed loss between the two embeddings.
    """

    assert mode in ['cosine', 'dot', 'euclidean'], "Invalid mode! Choose from 'cosine', 'dot', or 'euclidean'."
    assert reduction in ['mean', 'sum'], "Invalid reduction! Choose from 'mean' or 'sum'."

    if mode == 'cosine':
        # Use PyTorch's built-in cosine similarity function
        loss = 1 - F.cosine_similarity(embedding1, embedding2, dim=1)  # Shape (B, H, W)

    elif mode == 'dot':
        # Dot product similarity
        loss = -torch.sum(embedding1 * embedding2, dim=1)  # Shape (B, H, W)

    elif mode == 'euclidean':
        # L2 distance (Euclidean distance)
        loss = torch.norm(embedding1 - embedding2, dim=1, p=2)  # Shape (B, H, W)

    # Apply weight map if provided
    if weight is not None:
        loss = loss * weight

    # Reduce the loss
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()


def feature_distillation_loss(student_embedding, teacher_embedding, mode='cosine', reduction='mean', eps=1e-8):
    """
    Feature distillation loss: encourages student embeddings to match teacher embeddings.
    Uses numerically stable operations for bfloat16/float16 training.

    Args:
        student_embedding (Tensor): Student model embeddings of shape (B, C, H, W).
        teacher_embedding (Tensor): Teacher model embeddings of shape (B, C, H, W).
        mode (str): Similarity metric - 'cosine', 'mse', 'smooth_l1', or 'normalized_mse'.
        reduction (str): Reduction method - 'mean', 'sum', or 'none'.
        eps (float): Small constant for numerical stability.

    Returns:
        Tensor: The computed feature distillation loss.
    """
    assert mode in ['cosine', 'mse', 'smooth_l1', 'normalized_mse'], \
        f"Invalid mode '{mode}'. Choose from 'cosine', 'mse', 'smooth_l1', 'normalized_mse'."
    assert reduction in ['mean', 'sum', 'none'], \
        f"Invalid reduction '{reduction}'. Choose from 'mean', 'sum', 'none'."

    # Ensure inputs are float32 for stable computation
    student_f32 = student_embedding.float()
    teacher_f32 = teacher_embedding.float()

    if mode == 'cosine':
        # Cosine similarity loss: 1 - cos_sim
        # Normalize embeddings along channel dimension
        student_norm = F.normalize(student_f32, p=2, dim=1, eps=eps)
        teacher_norm = F.normalize(teacher_f32, p=2, dim=1, eps=eps)

        # Compute cosine similarity
        cos_sim = torch.sum(student_norm * teacher_norm, dim=1)  # Shape: (B, H, W)

        # Clamp to avoid numerical issues
        cos_sim = torch.clamp(cos_sim, -1.0 + eps, 1.0 - eps)

        # Loss is 1 - similarity (so similar embeddings have low loss)
        loss = 1.0 - cos_sim

    elif mode == 'mse':
        # Mean squared error between embeddings
        loss = torch.mean((student_f32 - teacher_f32) ** 2, dim=1)  # Shape: (B, H, W)

    elif mode == 'smooth_l1':
        # Smooth L1 loss (Huber loss) - more robust to outliers
        loss = F.smooth_l1_loss(student_f32, teacher_f32, reduction='none')
        loss = torch.mean(loss, dim=1)  # Average over channels

    elif mode == 'normalized_mse':
        # MSE on normalized embeddings - scale invariant
        student_norm = F.normalize(student_f32, p=2, dim=1, eps=eps)
        teacher_norm = F.normalize(teacher_f32, p=2, dim=1, eps=eps)
        loss = torch.mean((student_norm - teacher_norm) ** 2, dim=1)

    # Apply reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss




# CrossEntropyLoss with optional sigmoid and auto weight calculation
class CrossEntropyLoss(nn.Module):
    def __init__(self, auto_pos_weight=True, sigmoid=False, weight_clip_max=10.0, label_smoothing=0.0):
        """
        Cross-entropy loss with optional automatic class weighting and sigmoid activation.
        Includes numerical stability improvements for bfloat16/float16 training.

        Parameters:
        - auto_pos_weight (bool): Automatically calculate class weights based on target distribution.
        - sigmoid (bool): Apply sigmoid activation to predictions for binary classification.
        - weight_clip_max (float): Maximum value to clip class weights (prevents extreme weights).
        - label_smoothing (float): Label smoothing factor for regularization (0.0 = no smoothing).
        """
        super(CrossEntropyLoss, self).__init__()
        self.auto_pos_weight = auto_pos_weight
        self.sigmoid = sigmoid
        self.weight_clip_max = weight_clip_max
        self.label_smoothing = label_smoothing

    def forward(self, preds, targets):
        # Ensure preds are float32 for stable computation
        preds_f32 = preds.float()

        # Apply sigmoid to predictions if specified for binary classification
        if self.sigmoid:
            preds_f32 = torch.sigmoid(preds_f32)

        # Automatically calculate class weights if enabled
        if self.auto_pos_weight:
            # Calculate weights for each class based on the target distribution
            num_class_1 = torch.sum(targets == 1).float()
            num_class_0 = torch.sum(targets == 0).float()

            # Avoid division by zero and extreme weights
            num_class_1 = torch.clamp(num_class_1, min=1.0)
            num_class_0 = torch.clamp(num_class_0, min=1.0)

            # Compute weights (inverse frequency)
            total = num_class_0 + num_class_1
            weight_class_0 = total / (2.0 * num_class_0)
            weight_class_1 = total / (2.0 * num_class_1)

            # Clip weights to prevent extreme values
            weight_class_0 = torch.clamp(weight_class_0, max=self.weight_clip_max)
            weight_class_1 = torch.clamp(weight_class_1, max=self.weight_clip_max)

            weight = torch.tensor([weight_class_0.item(), weight_class_1.item()],
                                  dtype=torch.float32, device=targets.device)
            criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=self.label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        return criterion(preds_f32, targets)



# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, sigmoid=False):
        super(DiceLoss, self).__init__()
        self.sigmoid = sigmoid
        self.criterion = monai_losses.DiceLoss(to_onehot_y=True, sigmoid=False)

    def forward(self, preds, targets):
        if self.sigmoid:
            preds = torch.sigmoid(preds)
        return self.criterion(preds, targets.unsqueeze(1).long())

