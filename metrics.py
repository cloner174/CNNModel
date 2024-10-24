import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def calculate_segmentation_metrics(preds, masks, num_classes):
    """
    Calculate Dice Coefficient and IoU for segmentation.
    preds: Tensor of shape [B, H, W]
    masks: Tensor of shape [B, H, W]
    """
    preds = preds.cpu().numpy()
    masks = masks.cpu().numpy()
    dice = 0.0
    iou = 0.0
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        true_cls = (masks == cls)
        intersection = np.logical_and(pred_cls, true_cls).sum()
        union = np.logical_or(pred_cls, true_cls).sum()
        if union == 0:
            continue
        dice += (2. * intersection) / (np.sum(pred_cls) + np.sum(true_cls) + 1e-6)
        iou += intersection / (union + 1e-6)
    dice /= num_classes
    iou /= num_classes
    
    return dice, iou


def calculate_classification_metrics(preds, labels):
    """
    Calculate Precision, Recall, F1 Score, ROC-AUC for classification.
    preds: Tensor of shape [B, 1]
    labels: Tensor of shape [B, 1]
    """
    preds = torch.sigmoid(preds).cpu().numpy()
    labels = labels.cpu().numpy()
    preds_binary = (preds > 0.5).astype(int)
    precision = precision_score(labels, preds_binary, average='binary', zero_division=0)
    recall = recall_score(labels, preds_binary, average='binary', zero_division=0)
    f1 = f1_score(labels, preds_binary, average='binary', zero_division=0)
    roc_auc = roc_auc_score(labels, preds)
    return precision, recall, f1, roc_auc


def calculate_localization_metrics(preds, boxes):
    """
    Calculate Mean Absolute Error (MAE) for localization.
    preds: Tensor of shape [B, bbox_size]
    boxes: Tensor of shape [B, bbox_size]
    """
    mae = F.l1_loss(preds, boxes, reduction='mean').item()
    return mae

#cloner174