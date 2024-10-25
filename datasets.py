import torch
from torch.utils.data import Dataset

class NoduleDataset(Dataset):
    def __init__(self, images, masks=None, boxes=None, labels=None, transform=None):
        """
        images: shape [N, H, W]
        masks: shape [N, H, W]
        boxes:  [N, bbox_size]
        labels: [N] with binary classification labels
        transform: Optional torchvision transforms
        """
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
            mask = torch.tensor(self.masks[idx], dtype=torch.long)  # [H, W]
            box = torch.tensor(self.boxes[idx], dtype=torch.float32)  # [bbox_size]
            label = torch.tensor(self.labels[idx], dtype=torch.float32)  # Scalar
            
            box = box / 64.0 
            
            return image, mask, label, box
        
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)  # Scalar
            return image, label, idx
    
#cloner174