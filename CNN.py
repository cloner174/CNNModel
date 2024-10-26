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
    def __init__(self, input_channels=1, num_classes=2, bbox_size=4):
        super(MultitaskCNN_Pretrained, self).__init__()
        
        try:
            import torch
            weight_path = './weights/resnet34.pth'
            self.encoder = models.resnet34(weights=None)
            state_dict = torch.load(weight_path, weights_only=True)
            self.encoder.load_state_dict(state_dict)
        except:
            from torchvision.models import ResNet34_Weights
            self.encoder = models.resnet34(weights= ResNet34_Weights.IMAGENET1K_V1)
        
        if input_channels != 3:
            self.encoder.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.layer0 = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu,
            self.encoder.maxpool
        )
        self.layer1 = self.encoder.layer1  # 64 channels
        self.layer2 = self.encoder.layer2  # 128 channels
        self.layer3 = self.encoder.layer3  # 256 channels
        self.layer4 = self.encoder.layer4  # 512 channels
        
        # Updated AttentionBlock initializations
        self.att1 = AttentionBlock(F_g=256, F_l=256, F_int=256)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=128)
        self.att3 = AttentionBlock(F_g=64, F_l=64, F_int=64)
        
        self.segmentation_head = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            self.att1,  # AttentionBlock with F_g=256
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            self.att2,  # AttentionBlock with F_g=128
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            self.att3,  # AttentionBlock with F_g=64
            nn.Conv2d(64, num_classes, kernel_size=1)
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
        x0 = self.layer0(x) 
        x1 = self.layer1(x0) 
        x2 = self.layer2(x1)
        x3 = self.layer3(x2) 
        x4 = self.layer4(x3) 
        
        seg_out = self.segmentation_head[0](x4)  # ConvTranspose2d
        seg_out = self.segmentation_head[1](seg_out)  # BatchNorm
        seg_out = self.segmentation_head[2](seg_out)  # ReLU
        seg_out = self.segmentation_head[3](seg_out, x3)  # AttentionBlock
        
        seg_out = self.segmentation_head[4](seg_out)  # ConvTranspose2d
        seg_out = self.segmentation_head[5](seg_out)  # BatchNorm
        seg_out = self.segmentation_head[6](seg_out)  # ReLU
        seg_out = self.segmentation_head[7](seg_out, x2)  # AttentionBlock
        
        seg_out = self.segmentation_head[8](seg_out)  # ConvTranspose2d
        seg_out = self.segmentation_head[9](seg_out)  # BatchNorm
        seg_out = self.segmentation_head[10](seg_out)  # ReLU
        seg_out = self.segmentation_head[11](seg_out, x1)  # AttentionBlock
        
        seg_out = self.segmentation_head[12](seg_out)  # Conv2d
        seg_out = F.interpolate(seg_out, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        cls_out = self.classification_head(x4)
        
        loc_out = self.localization_head(x4)
        
        return seg_out, cls_out, loc_out



class MultitaskCNN(nn.Module):
    
    def __init__(self, input_channels=1, num_classes=2, bbox_size=4):
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
    
#cloner174