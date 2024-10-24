import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from datasets import FullyAnnotatedDataset, WeaklyAnnotatedDataset
from main import train_model
from CNN import MultitaskCNN
from utils import visualize_predictions


def go(base_dir):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    X = np.load(os.path.join(base_dir, 'X.npy'))            # [N, H, W]
    masks = np.load(os.path.join(base_dir, 'masks.npy'))    # [N, H, W] with integer class labels or -1 for weak
    boxes = np.load(os.path.join(base_dir, 'boxes.npy'))    # [N, bbox_size]
    y_class = np.load(os.path.join(base_dir, 'y_class.npy'))  # [N]
    
    annotated_indices = np.where(masks != -1)[0]
    weakly_annotated_indices = np.where(masks == -1)[0]
    
    X_annotated = X[annotated_indices]
    masks_annotated = masks[annotated_indices]
    boxes_annotated = boxes[annotated_indices]
    y_train_annotated = y_class[annotated_indices]
    
    X_weak = X[weakly_annotated_indices]
    y_train_weak = y_class[weakly_annotated_indices]
    
    X_train, X_val, masks_train, masks_val, boxes_train, boxes_val, y_train, y_val = train_test_split(
        X_annotated, masks_annotated, boxes_annotated, y_train_annotated,
        test_size=0.2, random_state=42
    )
    
    annotated_dataset = FullyAnnotatedDataset(X_train, masks_train, boxes_train, y_train, transform=None)
    weak_dataset = WeaklyAnnotatedDataset(X_weak, y_train_weak, transform=None)
    val_dataset = FullyAnnotatedDataset(X_val, masks_val, boxes_val, y_val, transform=None)
    
    annotated_loader = DataLoader(annotated_dataset, batch_size=32, shuffle=True, num_workers=4)
    weak_loader = DataLoader(weak_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    model = MultitaskCNN(input_channels=1, num_classes=13, bbox_size=4).to(device)
    
    train_model(model, annotated_loader, weak_loader, val_loader, device, num_epochs=50, patience=10, base_dir=base_dir)
    
    model.load_state_dict(torch.load(os.path.join(base_dir, 'best_model.pth')))
    model.to(device)
    
    visualize_predictions(model, val_loader, device, num_samples=5, num_classes=13)


if __name__ == "__main__":
    bd = input('Please Insert The path to main dir of data: ')
    go(bd)