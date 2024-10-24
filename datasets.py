import torch
from torch.utils.data import Dataset


class FullyAnnotatedDataset(Dataset):
    
    def __init__(self, images, masks, boxes, labels, transform=None):
        """
        images: NumPy array of shape [N, H, W]
        masks: NumPy array of shape [N, H, W] with integer class labels
        boxes: NumPy array of shape [N, bbox_size]
        labels: NumPy array of shape [N] with binary classification labels
        transform: Optional torchvision transforms
        """
        self.images = images
        self.masks = masks
        self.boxes = boxes
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]  
        mask = self.masks[idx] 
        box = self.boxes[idx] 
        label = self.labels[idx]  
        
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) 
        mask = torch.tensor(mask, dtype=torch.long) 
        box = torch.tensor(box, dtype=torch.float32) 
        label = torch.tensor(label, dtype=torch.float32) 
        
        box = box / 64.0
        
        if self.transform:
            image = self.transform(image)
        
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
        image = self.images[idx]  
        label = self.labels[idx] 
        
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) 
        label = torch.tensor(label, dtype=torch.float32)   
        
        if self.transform:
            image = self.transform(image)
        
        assert torch.isfinite(image).all(), "Image contains NaN or Inf."
        assert torch.isfinite(label).all(), "Label contains NaN or Inf."
        
        return image, label
    
#cloner174