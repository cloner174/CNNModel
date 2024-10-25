import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from metrics import calculate_segmentation_metrics, calculate_classification_metrics, calculate_localization_metrics
from torch.cuda.amp import autocast, GradScaler

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        self.log_sigma1 = nn.Parameter(torch.zeros(1))  # Segmentation
        self.log_sigma2 = nn.Parameter(torch.zeros(1))  # Classification
        self.log_sigma3 = nn.Parameter(torch.zeros(1))  # Localization

    def forward(self, loss_seg, loss_cls, loss_loc):
        loss = (torch.exp(-self.log_sigma1) * loss_seg + self.log_sigma1) + \
               (torch.exp(-self.log_sigma2) * loss_cls + self.log_sigma2) + \
               (torch.exp(-self.log_sigma3) * loss_loc + self.log_sigma3)
        return loss

def train_model(model, train_loader, val_loader, device, num_epochs=50, patience=10, base_dir='./'):
    criterion_seg = nn.CrossEntropyLoss()
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_loc = nn.SmoothL1Loss()
    multi_task_loss = MultiTaskLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    scaler = GradScaler()
    best_val_loss = float('inf')
    counter = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_seg_loss = 0.0
        train_cls_loss = 0.0
        train_loc_loss = 0.0
        train_total = 0
        for (batch_a, batch_w) in tqdm(zip(train_loader['annotated'], train_loader['weak']), desc=f'Epoch {epoch}/{num_epochs} - Training', total=min(len(train_loader['annotated']), len(train_loader['weak']))):
            images_a, masks_a, labels_a, boxes_a = batch_a
            images_a = images_a.to(device)
            masks_a = masks_a.to(device)
            boxes_a = boxes_a.to(device)
            labels_a = labels_a.to(device).unsqueeze(1)
            images_w, labels_w, idx_w = batch_w
            images_w = images_w.to(device)
            labels_w = labels_w.to(device).unsqueeze(1)
            optimizer.zero_grad()
            with autocast():
                outputs_seg_a, outputs_cls_a, outputs_loc_a = model(images_a)
                loss_seg_a = criterion_seg(outputs_seg_a, masks_a)
                loss_cls_a = criterion_cls(outputs_cls_a, labels_a)
                loss_loc_a = criterion_loc(outputs_loc_a, boxes_a)
                outputs_seg_w, outputs_cls_w, outputs_loc_w = model(images_w)
                loss_cls_w = criterion_cls(outputs_cls_w, labels_w)
                probs_seg_w = torch.softmax(outputs_seg_w, dim=1)
                max_probs, pseudo_masks_w = torch.max(probs_seg_w, dim=1)
                confidence_threshold = 0.8
                confident_mask = max_probs > confidence_threshold
                pseudo_masks_w = pseudo_masks_w * confident_mask.long()
                if confident_mask.sum() > 0:
                    loss_seg_w = criterion_seg(outputs_seg_w, pseudo_masks_w) * confident_mask.float().mean()
                else:
                    loss_seg_w = 0.0
                
                loss_a = multi_task_loss(loss_seg_a, loss_cls_a, loss_loc_a)
                loss_w = loss_cls_w + loss_seg_w
                total_loss = loss_a + loss_w
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_seg_loss += (loss_seg_a.item() + loss_seg_w.item()) * images_a.size(0)
            train_cls_loss += (loss_cls_a.item() + loss_cls_w.item()) * images_a.size(0)
            train_loc_loss += loss_loc_a.item() * images_a.size(0)
            train_total += images_a.size(0)
        
        avg_train_seg_loss = train_seg_loss / train_total
        avg_train_cls_loss = train_cls_loss / train_total
        avg_train_loc_loss = train_loc_loss / train_total
        avg_train_loss = avg_train_seg_loss + avg_train_cls_loss + avg_train_loc_loss
        
        model.eval()
        val_seg_loss = 0.0
        val_cls_loss = 0.0
        val_loc_loss = 0.0
        val_total = 0
        correct_cls = 0
        total_cls = 0
        dice_score = 0.0
        iou_score = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        roc_auc = 0.0
        mae = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch}/{num_epochs} - Validation'):
                images, masks, labels, boxes  = batch
                images = images.to(device)
                masks = masks.to(device)
                boxes = boxes.to(device)
                labels = labels.to(device).unsqueeze(1)
                outputs_seg, outputs_cls, outputs_loc = model(images)
                loss_seg_val = criterion_seg(outputs_seg, masks)
                loss_cls_val = criterion_cls(outputs_cls, labels)
                loss_loc_val = criterion_loc(outputs_loc, boxes)
                val_seg_loss += loss_seg_val.item() * images.size(0)
                val_cls_loss += loss_cls_val.item() * images.size(0)
                val_loc_loss += loss_loc_val.item() * images.size(0)
                val_total += images.size(0)
                preds_cls = (torch.sigmoid(outputs_cls) > 0.5).float()
                correct_cls += (preds_cls == labels).sum().item()
                total_cls += labels.size(0)
                dice, iou = calculate_segmentation_metrics(torch.argmax(outputs_seg, dim=1), masks, num_classes=13)
                dice_score += dice * images.size(0)
                iou_score += iou * images.size(0)
                prec, rec, f1_score_val, roc_auc_val = calculate_classification_metrics(outputs_cls, labels)
                precision += prec * images.size(0)
                recall += rec * images.size(0)
                f1 += f1_score_val * images.size(0)
                roc_auc += roc_auc_val * images.size(0)
                mae_val = calculate_localization_metrics(outputs_loc, boxes)
                mae += mae_val * images.size(0)
        
        avg_val_seg_loss = val_seg_loss / val_total
        avg_val_cls_loss = val_cls_loss / val_total
        avg_val_loc_loss = val_loc_loss / val_total
        avg_val_loss = avg_val_seg_loss + avg_val_cls_loss + avg_val_loc_loss
        val_accuracy = correct_cls / total_cls
        dice_score /= val_total
        iou_score /= val_total
        precision /= val_total
        recall /= val_total
        f1 /= val_total
        roc_auc /= val_total
        mae /= val_total
        scheduler.step(avg_val_loss)
        print(f'Epoch [{epoch}/{num_epochs}] '
              f'Train Loss: {avg_train_loss:.4f} (Seg: {avg_train_seg_loss:.4f}, '
              f'Cls: {avg_train_cls_loss:.4f}, Loc: {avg_train_loc_loss:.4f}) '
              f'Val Loss: {avg_val_loss:.4f} (Seg: {avg_val_seg_loss:.4f}, '
              f'Cls: {avg_val_cls_loss:.4f}, Loc: {avg_val_loc_loss:.4f}) '
              f'Val Acc: {val_accuracy:.4f} '
              f'Dice: {dice_score:.4f} IoU: {iou_score:.4f} '
              f'Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f} '
              f'ROC-AUC: {roc_auc:.4f} MAE: {mae:.4f}')
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(base_dir, 'best_model.pth'))
            print('Best model saved!')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break
    
#cloner174