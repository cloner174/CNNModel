import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

class NoduleDataset(Dataset):
    def __init__(self, images, masks=None, boxes=None, labels=None, transform=None):
        self.images = images
        self.masks = masks
        self.boxes = boxes
        self.labels = labels
        self.transform = transform
        
        if self.masks is not None and self.boxes is not None and self.labels is not None:
            self.is_fully_annotated = True
        else:
            self.is_fully_annotated = False

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        if self.transform:
            image = self.transform(image)

        if self.is_fully_annotated:
            
            mask = torch.tensor(self.masks[idx], dtype=torch.long).unsqueeze(0)  # [H, W]
            box = torch.tensor(self.boxes[idx], dtype=torch.float32).view(-1)  # [bbox_size]
            label = torch.tensor(self.labels[idx], dtype=torch.float32)  # Scalar
            
            box = box / 64.0 
            
            return image, mask, label, box
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)  # Scalar
            
            return image, label, idx 
    


class FullyAnnotatedDataset(Dataset):
    
    def __init__(self, images, masks, boxes, labels, transform=None, set_boxes_to_label = False):
        """
        images: NumPy array of shape [N, H, W]
        masks: NumPy array of shape [N, H, W] with integer class labels or -1 for weakly annotated
        boxes: NumPy array of shape [N, bbox_size]
        labels: NumPy array of shape [N] with binary classification labels
        transform: Optional torchvision transforms
        """
        self.images = images
        self.masks = masks
        self.boxes = boxes
        self.labels = labels
        self.transform = transform
        
        if set_boxes_to_label:
            self.boxes[self.labels == 0] = 0.0
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]       # [H, W]
        mask = self.masks[idx]         # [H, W]
        box = self.boxes[idx]          # [bbox_size]
        label = self.labels[idx]       # scalar
        
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        mask = torch.tensor(mask, dtype=torch.long)                   # [H, W]
        box = torch.tensor(box, dtype=torch.float32).view(-1)                    # [bbox_size]
        label = torch.tensor(label, dtype=torch.float32)               # [1]
        
        box = box / 64.0
        
        if self.transform:
            augmented = self.transform(image)
            image = augmented
        
        assert torch.isfinite(image).all(), "Image contains NaN or Inf."
        assert torch.isfinite(mask).all(), "Mask contains NaN or Inf."
        assert torch.isfinite(box).all(), "Box contains NaN or Inf."
        assert torch.isfinite(label).all(), "Label contains NaN or Inf."
        
        return image, mask, label, box



class WeaklyAnnotatedDataset(Dataset):
    
    def __init__(self, images, labels, transform=None):
        """
        images: NumPy array of shape [N, H, W]
        labels: NumPy array of shape [N] with binary classification labels
        transform: Optional torchvision transforms
        """
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]       # [H, W]
        label = self.labels[idx]       # scalar
        
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        label = torch.tensor(label, dtype=torch.float32)               # [1]
        
        if self.transform:
            augmented = self.transform(image)
            image = augmented
        
        assert torch.isfinite(image).all(), "Image contains NaN or Inf."
        assert torch.isfinite(label).all(), "Label contains NaN or Inf."
        
        return image, label
    
#cloner174