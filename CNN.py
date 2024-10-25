from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

class CNNBackbone(nn.Module):
    def __init__(self, input_channels=1):
        super(CNNBackbone, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)) 
        )
    
    def forward(self, x):
        features = self.features(x)
        return features

class MultitaskCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=13, bbox_size=4):
        super(MultitaskCNN, self).__init__()
        self.backbone = CNNBackbone(input_channels)
        
        self.segmentation_head = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1) 
        )
        
        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        
        self.localization_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, bbox_size) 
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        seg_out = self.segmentation_head(features)
        seg_out = F.interpolate(seg_out, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        cls_out = self.classification_head(features)
        
        loc_out = self.localization_head(features)
        
        return seg_out, cls_out, loc_out

# CNN.py

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class MultitaskCNN_Pretrained(nn.Module):
    def __init__(self, input_channels=1, num_classes=13, bbox_size=4):
        super(MultitaskCNN_Pretrained, self).__init__()
        self.encoder = models.resnet34(pretrained=True)
        
        if input_channels != 3:
            self.encoder.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        
        self.segmentation_head = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        
        self.localization_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, bbox_size)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        
        seg_out = self.segmentation_head(features)
        seg_out = F.interpolate(seg_out, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        cls_out = self.classification_head(features)
        
        loc_out = self.localization_head(features)
        
        return seg_out, cls_out, loc_out

#cloner174