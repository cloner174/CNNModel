import torch
from torch.utils.data import DataLoader
from datasets import NoduleDataset
from CNN import MultitaskCNN_Pretrained
from main import train_model
from utils import visualize_predictions
from metrics import calculate_segmentation_metrics, calculate_classification_metrics, calculate_localization_metrics
import torchvision.transforms as transforms
import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import os
from PIL import Image
from sklearn.model_selection import train_test_split

torch.manual_seed(42)
np.random.seed(42)

def go(annotated_images,
       annotated_masks,
       annotated_labels,
       annotated_boxes,
       weak_images,
       weak_labels):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    annotated_images, val_images, annotated_masks, val_masks, annotated_boxes, val_boxes, annotated_labels, val_labels,weak_images, val_masks, weak_labels, val_labels = train_test_split(
        annotated_images,annotated_masks,annotated_labels,annotated_boxes,weak_images,weak_labels,
        test_size=0.2, random_state=42
    )
    
    annotated_dataset = NoduleDataset(images=annotated_images, masks=annotated_masks, boxes=annotated_boxes, labels=annotated_labels, transform=None)
    weak_dataset = NoduleDataset(images=weak_images, labels=weak_labels, transform=None)
    val_dataset = NoduleDataset(images=val_images, masks=val_masks, boxes=val_boxes, labels=val_labels, transform=None)
    annotated_loader = DataLoader(annotated_dataset, batch_size=16, shuffle=True, num_workers=4)
    weak_loader = DataLoader(weak_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    train_loader = {'annotated': annotated_loader, 'weak': weak_loader}
    model = MultitaskCNN_Pretrained(input_channels=1, num_classes=13, bbox_size=4)
    model = model.to(device)
    train_model(model, train_loader, val_loader, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), num_epochs=50, patience=10, base_dir='./')
    visualize_predictions(model, val_loader, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), num_samples=5, num_classes=13)
    

if __name__ == "__main__":
    bd = input('Please Insert The path to main dir of data: ')
    if len(bd) < 5 : 
        bd = './Final_Step/one'
    
    X = []
    masks = []
    y_class = []
    y_centroids = []
    proccessed = []
    for any_image in os.listdir(bd):
        if any_image.endswith('.png'):
            name = any_image.split('_label_1')[0] + '_label_1'
            if name not in proccessed :
                proccessed.append(name)
                image_path = os.path.join(bd, name + '.png')
                mask_path = os.path.join(bd, name + '_mask.png')
                centroid_path = os.path.join(bd, name + '.npy')
                image = Image.open(image_path).convert('L')
                mask = Image.open(image_path)
                centroid = np.load(centroid_path)
                X.append(np.array(image))
                masks.append(mask)
                y_centroids.append(centroid)
                y_class.append(1)
    
    del proccessed
    
    
    X = np.array(X)
    masks = np.array(masks)
    y_class = np.array(y_class)
    y_centroids = np.array(y_centroids)
    
    
    print('done! X.shape , y_class.shape , masks.shape , y_centroids.shape -> ',X.shape , y_class.shape , masks.shape , y_centroids.shape)
    
    images = []
    for any_image in os.listdir('./generated_samples109/luna16'):
        image_path = os.path.join('./generated_samples109/luna16', any_image)
        image = Image.open(image_path).convert('L')
        images.append(np.array(image))
    
    labels = [ 1 for i in range(len(images))]
    images = np.array(images)
    labels = np.array(labels)
    print('done! images.shape, labels.shape -> ',images.shape, labels.shape)
    go(X, masks,y_class,y_centroids,images,labels)
    
#cloner174