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
    def __init__(self):
        super(WeightedMSE, self).__init__()

    def forward(self, pred, target, weight=None):
        s1 = torch.prod(torch.tensor(pred.size()[2:]).float())
        s2 = pred.size()[0]
        norm_term = (s1 * s2).cuda()

        if weight is None:
            loss = torch.sum((pred - target) ** 2) / norm_term
        else:
            loss = torch.sum(weight * (pred - target) ** 2) / norm_term

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
    #print(f"target shape: {target.shape}, weightmap shape: {weightmap.shape}, mask shape: {mask.shape}")

    if mode == 'AAAI':
        embedding = F.normalize(embedding, p=2, dim=1)
    mask = mask.float()

    affs = torch.zeros_like(target)
    loss = torch.tensor(0.0, dtype=embedding.dtype, device=embedding.device)
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
        loss_temp, affs_temp = embedding_single_offset_loss(embedding, shift_off, target[:,i], weightmap[:,i], mask[:,i], criterion, mode=mode)
        # loss += loss_temp * affs0_weight_factor[i]
        loss += loss_temp
        # all_loss.append((loss_temp * affs0_weight_factor[i]).item())
        all_loss.append(loss_temp.item())
        # if i < 2:
        #     loss += loss_temp * affs0_weight
        # else:
        #     loss += loss_temp
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


# CrossEntropyLoss with optional sigmoid and auto weight calculation
class CrossEntropyLoss(nn.Module):
    def __init__(self, auto_pos_weight=True, sigmoid=False):
        """
        Cross-entropy loss with optional automatic class weighting and sigmoid activation.

        Parameters:
        - auto_pos_weight (bool): Automatically calculate class weights based on target distribution.
        - sigmoid (bool): Apply sigmoid activation to predictions for binary classification.
        """
        super(CrossEntropyLoss, self).__init__()
        self.auto_pos_weight = auto_pos_weight
        self.sigmoid = sigmoid

    def forward(self, preds, targets):
        # Apply sigmoid to predictions if specified for binary classification
        if self.sigmoid:
            preds = torch.sigmoid(preds)

        # Automatically calculate class weights if enabled
        if self.auto_pos_weight:
            # Calculate weights for each class based on the target distribution
            weight = torch.tensor([torch.sum(targets == 1).item(), torch.sum(targets == 0).item()], dtype=torch.float32).to(targets.device)
            criterion = nn.CrossEntropyLoss(weight=weight)
        else:
            criterion = nn.CrossEntropyLoss()

        return criterion(preds, targets)



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

