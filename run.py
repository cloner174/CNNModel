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
    
    # ====== Load Data ======
    
    #annotated_images = np.random.rand(100, 256, 256)  # [N, H, W]
    #annotated_masks = np.random.randint(0, 13, (100, 256, 256))  # [N, H, W]
    #annotated_boxes = np.random.rand(100, 4)  # [N, bbox_size]
    #annotated_labels = np.random.randint(0, 2, 100)  # [N]
    
    #weak_images = np.random.rand(50, 256, 256)  # [N, H, W]
    #weak_labels = np.random.randint(0, 2, 50)  # [N]
    
    #val_images = np.random.rand(20, 256, 256)  # [N, H, W]
    #val_masks = np.random.randint(0, 13, (20, 256, 256))  # [N, H, W]
    #val_boxes = np.random.rand(20, 4)  # [N, bbox_size]
    #val_labels = np.random.randint(0, 2, 20)  # [N]
    
    # ====== Transformations ======
    #train_transform = transforms.Compose([
    #    transforms.RandomHorizontalFlip(),
    #    transforms.RandomRotation(10),
    #    transforms.Normalize(mean=[0.5], std=[0.5]),
    #])
    
    #val_transform = transforms.Compose([
    #    transforms.Normalize(mean=[0.5], std=[0.5]),
    #])
    train_transform = None
    val_transform = None
    
    
    annotated_images, val_images, annotated_masks, val_masks, annotated_boxes, val_boxes, annotated_labels, val_labels,weak_images, val_masks, weak_labels, val_labels = train_test_split(
        annotated_images,annotated_masks,annotated_labels,annotated_boxes,weak_images,weak_labels,
        test_size=0.2, random_state=42
    )
    # ====== Dataset Instances ======
    annotated_dataset = NoduleDataset(images=annotated_images, masks=annotated_masks, 
                                      boxes=annotated_boxes, labels=annotated_labels, 
                                      transform=train_transform)
    weak_dataset = NoduleDataset(images=weak_images, labels=weak_labels, 
                                 transform=train_transform)
    val_dataset = NoduleDataset(images=val_images, masks=val_masks, 
                                boxes=val_boxes, labels=val_labels, 
                                transform=val_transform)
    
    # ====== DataLoaders ======
    def get_dataloader(dataset, batch_size, shuffle, num_workers=4):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    # will be overridden by PSO
    initial_batch_size = 16
    
    annotated_loader = get_dataloader(annotated_dataset, batch_size=initial_batch_size, shuffle=True)
    weak_loader = get_dataloader(weak_dataset, batch_size=initial_batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=initial_batch_size, shuffle=False)
    
    train_loader = {'annotated': annotated_loader, 'weak': weak_loader}
    
    # ====== Initialize Model ======
    model = MultitaskCNN_Pretrained(input_channels=1, num_classes=13, bbox_size=4)
    model = model.to(device)
    
    # ====== Objective Functions for PSO ======
    
    def hyperparameter_objective_function(hyperparams):
        """
        hyperparams: array of shape (n_particles, dimensions)
                     Each row corresponds to a particle's hyperparameters.
                     Example dimensions: [learning_rate, batch_size]
        Returns:
            fitness: array of shape (n_particles,)
        """
        fitness = []
        for params in hyperparams:
            learning_rate = params[0]
            batch_size = int(params[1])
            
            # new DataLoaders with current batch_size
            annotated_loader_pso = get_dataloader(annotated_dataset, batch_size=batch_size, shuffle=True)
            weak_loader_pso = get_dataloader(weak_dataset, batch_size=batch_size, shuffle=True)
            val_loader_pso = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
            train_loader_pso = {'annotated': annotated_loader_pso, 'weak': weak_loader_pso}
            
            # a fresh model for each hyperparameter set
            model_pso = MultitaskCNN_Pretrained(input_channels=1, num_classes=13, bbox_size=4)
            model_pso = model_pso.to(device)
            
            # loss weights (fixed during hyperparameter optimization)
            loss_weights = {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0}
            
            # Train the model
            train_model(model_pso, train_loader_pso, val_loader_pso, device, 
                       hyperparams={'learning_rate': learning_rate}, 
                       loss_weights=loss_weights, 
                       num_epochs=5, patience=2, base_dir='./')
            
            # Load best model
            best_model_path = os.path.join('./', 'best_model.pth')
            if os.path.exists(best_model_path):
                model_pso.load_state_dict(torch.load(best_model_path))
                model_pso.eval()
                
                # Evaluate
                dice_scores = []
                f1_scores = []
                with torch.no_grad():
                    for batch in val_loader_pso:
                        images, masks, boxes, labels = batch
                        images = images.to(device)
                        masks = masks.to(device)
                        labels = labels.to(device).unsqueeze(1)
                        
                        outputs_seg, outputs_cls, outputs_loc = model_pso(images)
                        preds_seg = torch.argmax(outputs_seg, dim=1)
                        dice, _ = calculate_segmentation_metrics(preds_seg, masks, num_classes=13)
                        dice_scores.append(dice)
                        
                        preds_cls = (torch.sigmoid(outputs_cls) > 0.5).float()
                        _, _, f1_score_val, _ = calculate_classification_metrics(outputs_cls, labels)
                        f1_scores.append(f1_score_val)
                
                avg_dice = np.mean(dice_scores)
                avg_f1 = np.mean(f1_scores)
                
                # PSO minimizes, so negate
                combined_metric = -(avg_dice + avg_f1)
                fitness.append(combined_metric)
            else:
                # training failed -> assign a high loss
                fitness.append(1e6)
        
        return np.array(fitness)
    
    
    def loss_weight_objective_function(loss_weights_pso):
        """
        loss_weights_pso: array of shape (n_particles, dimensions)
                          Each row corresponds to a particle's loss weights.
                          Example dimensions: [alpha, beta, gamma]
        Returns:
            fitness: array of shape (n_particles,)
        """
        fitness = []
        for weights in loss_weights_pso:
            alpha, beta, gamma = weights
            
            # fixed hyperparameters for loss weight tuning
            hyperparams = {'learning_rate': 1e-4}
            loss_weights = {'alpha': alpha, 'beta': beta, 'gamma': gamma}
            
            train_model(model, train_loader, val_loader, device, 
                       hyperparams=hyperparams, 
                       loss_weights=loss_weights, 
                       num_epochs=5, patience=2, base_dir='./')
            
            best_model_path = os.path.join('./', 'best_model.pth')
            if os.path.exists(best_model_path):
                model.load_state_dict(torch.load(best_model_path))
                model.eval()
                
                dice_scores = []
                f1_scores = []
                with torch.no_grad():
                    for batch in val_loader:
                        images, masks, boxes, labels = batch
                        images = images.to(device)
                        masks = masks.to(device)
                        labels = labels.to(device).unsqueeze(1)
                        
                        outputs_seg, outputs_cls, outputs_loc = model(images)
                        preds_seg = torch.argmax(outputs_seg, dim=1)
                        dice, _ = calculate_segmentation_metrics(preds_seg, masks, num_classes=13)
                        dice_scores.append(dice)
                        
                        preds_cls = (torch.sigmoid(outputs_cls) > 0.5).float()
                        _, _, f1_score_val, _ = calculate_classification_metrics(outputs_cls, labels)
                        f1_scores.append(f1_score_val)
                
                avg_dice = np.mean(dice_scores)
                avg_f1 = np.mean(f1_scores)
                
                combined_metric = -(avg_dice + avg_f1)
                fitness.append(combined_metric)
            else:
                fitness.append(1e6)
        
        return np.array(fitness)
    
    # ====== PSO for Hyperparameter Optimization ======
    
    # bounds for hyperparameters
    # learning_rate between 1e-5 and 1e-3
    # batch_size between 16 and 64
    lb_hyper = [1e-5, 16]
    ub_hyper = [1e-3, 64]
    
    # PSO optimizer for hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    optimizer_hyper = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=(lb_hyper, ub_hyper))
    
    best_cost_hyper, best_pos_hyper = optimizer_hyper.optimize(hyperparameter_objective_function, iters=10)
    
    print("Best Hyperparameters:")
    print(f"Learning Rate: {best_pos_hyper[0]:.6f}")
    print(f"Batch Size: {int(best_pos_hyper[1])}")
    
    # ====== Train Final Model with Optimized Hyperparameters ======
    
    # define optimized hyperparameters
    optimized_hyperparams = {'learning_rate': best_pos_hyper[0]}
    optimized_batch_size = int(best_pos_hyper[1])
    
    # update DataLoaders with optimized batch size
    annotated_loader_final = get_dataloader(annotated_dataset, batch_size=optimized_batch_size, shuffle=True)
    weak_loader_final = get_dataloader(weak_dataset, batch_size=optimized_batch_size, shuffle=True)
    val_loader_final = get_dataloader(val_dataset, batch_size=optimized_batch_size, shuffle=False)
    
    train_loader_final = {'annotated': annotated_loader_final, 'weak': weak_loader_final}
    
    # initialize a fresh model
    model_final = MultitaskCNN_Pretrained(input_channels=1, num_classes=13, bbox_size=4)
    model_final = model_final.to(device)
    
    # train the final model with optimized hyperparameters and default loss weights
    train_model(model_final, train_loader_final, val_loader_final, device, 
               hyperparams=optimized_hyperparams, 
               loss_weights={'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0}, 
               num_epochs=50, patience=10, base_dir='./')
    
    # ====== Define PSO for Loss Weight Tuning ======
    def loss_weight_objective_wrapper(loss_weights_pso):
        """
        Wrapper for loss weight objective to avoid redefining DataLoaders.
        """
        return loss_weight_objective_function(loss_weights_pso)
    
    lb_loss = [0.1, 0.1, 0.1]
    ub_loss = [2.0, 2.0, 2.0]
    
    optimizer_loss = ps.single.GlobalBestPSO(n_particles=10, dimensions=3, options=options, bounds=(lb_loss, ub_loss))
    
    best_cost_loss, best_pos_loss = optimizer_loss.optimize(loss_weight_objective_wrapper, iters=10)
    
    print("Best Loss Weights:")
    print(f"Alpha (Segmentation): {best_pos_loss[0]:.4f}")
    print(f"Beta (Classification): {best_pos_loss[1]:.4f}")
    print(f"Gamma (Localization): {best_pos_loss[2]:.4f}")
    
    # ====== Train Final Model with Optimized Loss Weights ======
    
    optimized_loss_weights = {'alpha': best_pos_loss[0], 
                              'beta': best_pos_loss[1], 
                              'gamma': best_pos_loss[2]}
    
    train_model(model_final, train_loader_final, val_loader_final, device, 
               hyperparams=optimized_hyperparams, 
               loss_weights=optimized_loss_weights, 
               num_epochs=50, patience=10, base_dir='./')
    
    # ====== Visualize Predictions ======
    visualize_predictions(model_final, val_loader_final, device, num_samples=5, num_classes=13)
    

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