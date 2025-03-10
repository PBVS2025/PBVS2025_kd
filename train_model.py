import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from probabilistic_classifier_resnet_cl import EO_SAR_Model
from eo_sar_data import SarEODataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F 
from datetime import datetime
import argparse
import logging
import sys
from utils import compute_adjustment
import torch.nn as nn
from loss import ContrastiveLoss, SupervisedContrastiveLoss, FeatureMatchingLoss
from torchsampler import ImbalancedDatasetSampler
import gc
import random
from torch.autograd import Variable

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for probabilistic classifier')
    parser.add_argument('--data_root', type=str, required=True,
                      help='Root directory of the dataset')
    parser.add_argument('--output_dir', type=str, default='experiments',
                      help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs to train')
    parser.add_argument('--val_iter', type=int, default=1,
                      help='Validation interval')
    
    parser.add_argument('--adj_dir', type=str, help="Adjustment Directory")
    parser.add_argument('--use_adjustments', action='store_true', 
                      help='Whether to use logit adjustments')
    parser.add_argument('--pretrained_path', type=str, default=None,
                      help='Path to pretrained weights')
    parser.add_argument('--use_class_weights', action='store_true', help='Use class weights')
    
    parser.add_argument('--use_contrastive', action='store_true', help='Use contrastive learning')
    parser.add_argument('--contrastive_weight', type=float, default=0.1, help='Weight for contrastive loss')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for contrastive loss')

    parser.add_argument('--optimizer', type=str, default='AdamW',
                      help='Optimizer to use (AdamW or SGD)')
    parser.add_argument('--lr', type=float, default=5e-4,
                      help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                      help='Weight decay for optimizer')
    parser.add_argument('--gpus', type=str, default='0',
                      help='Comma-separated list of GPU device numbers to use (e.g., "0,1,2")')

    parser.add_argument('--contrastive_method', type=str, default='none',
                      choices=['none', 'supcon', 'simclr'],
                      help='Contrastive learning method to use (none | supcon | simclr)')

    parser.add_argument('--use_feature_matching', action='store_true', help='Use feature matching loss')
    parser.add_argument('--feature_matching_weight', type=float, default=0.1, help='Weight for feature matching loss')
    parser.add_argument('--moving_average_decay', type=float, default=0.9, help='Moving average decay for feature matching loss')
    parser.add_argument('--confidence_lambda', type=float, default=1.0, help='Weight for confidence loss')
    parser.add_argument('--use_confidence', action='store_true', help='Use confidence prediction')

    return parser.parse_args()

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()
def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_adjustment(adjustment, file_path):
    torch.save(adjustment, file_path)
    print(f"로짓 조정값이 {file_path}에 저장되었습니다.")

def load_adjustment(file_path, device):
    if os.path.exists(file_path):
        adjustment = torch.load(file_path, map_location=device)
        print(f"로짓 조정값을 {file_path}에서 불러왔습니다.")
        return adjustment
    else:
        print(f"파일 {file_path}이 존재하지 않습니다.")
        return None
    
def encode_onehot(labels, n_classes):
    onehot = torch.FloatTensor(labels.size()[0], n_classes)
    labels = labels.data
    if labels.is_cuda:
        onehot = onehot.cuda()
    onehot.zero_()
    onehot.scatter_(1, labels.view(-1, 1), 1)
    return onehot

def main():
    args = parse_args()
    
    # Set random seed
    set_seed(42)
    
    # Add these lines
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adjustment = None
    if args.use_adjustments and args.adj_dir:
        adjustment = load_adjustment(args.adj_dir, device)

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(args.output_dir, current_time)
    os.makedirs(exp_dir, exist_ok=True)
    logger = setup_logging(exp_dir)
    
    writer = SummaryWriter(exp_dir)
    
    logger.info(f"Configuration:")
    logger.info(f"- Data root: {args.data_root}")
    logger.info(f"- Output directory: {exp_dir}")
    logger.info(f"- Batch size: {args.batch_size}")
    logger.info(f"- Epochs: {args.epochs}")
    logger.info(f"- Device: {device}")

    train_eo_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.2892, 0.2892, 0.2892],
            std=[0.1455, 0.1455, 0.1455]
        )
    ])

    train_sar_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3)
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4058, 0.4058, 0.4058],
            std=[0.1282, 0.1282, 0.1282]
        )
    ])

    val_eo_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.2892, 0.2892, 0.2892],
            std=[0.1455, 0.1455, 0.1455]
        )
    ])

    val_sar_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4058, 0.4058, 0.4058],
            std=[0.1282, 0.1282, 0.1282]
        )
    ])

    logger.info("Loading training dataset...")
    train_dataset = SarEODataset(
        root_dir=os.path.join(args.data_root, "train"),
        sar_transform=train_sar_transform,
        eo_transform=train_eo_transform,
        use_contrastive=(args.contrastive_method != 'none')  # contrastive learning 사용 여부 전달
    )
    
    logger.info("Loading validation dataset...")
    val_dataset = SarEODataset(
        root_dir=os.path.join(args.data_root, "val"),
        sar_transform=val_sar_transform,
        eo_transform=val_eo_transform,
        use_contrastive=False  # validation에서는 contrastive learning 사용 안 함
    )

    real_class_counts = {
        "sedan": 364291,
        "SUV": 43401,
        "pickup_truck": 24158,
        "van": 16890,
        "box_truck": 2896,
        "motorcycle": 1441,
        "flatbed_truck": 898,
        "bus": 612,
        "pickup_truck_w_trailer": 695,
        "semi_w_trailer": 353
    }

    class_counts = [real_class_counts[class_name] for class_name in train_dataset.classes]
    logger.info(f"Real dataset class distribution: {class_counts}")
    
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    logger.info(f"Class weights: {class_weights}")

    train_loader = DataLoader(
        train_dataset,
        sampler=ImbalancedDatasetSampler(train_dataset),
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logger.info(f"Dataset sizes - Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    adjustment_path = os.path.join(args.output_dir, "adjustment")
    
    if args.use_adjustments and adjustment is None:
        adjustment_path = os.path.join(exp_dir, "adjustment.pt")
        adjustment = compute_adjustment(train_loader, tro=1.0, device=device)
        save_adjustment(adjustment, adjustment_path)
    
    model = EO_SAR_Model(
        num_classes=len(train_dataset.classes),
        eo_pretrained=args.pretrained_path,
        device=device,
        use_contrastive=(args.contrastive_method != 'none'),
        use_confidence=args.use_confidence 
    )
    
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    model.to(device)
    logger.info(f"Model created and moved to {device}")

    if args.optimizer.lower() == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        logger.info(f"Using AdamW optimizer with lr={args.lr}, weight_decay={args.weight_decay}")
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
        logger.info(f"Using SGD optimizer with lr={args.lr}, momentum=0.9, weight_decay={args.weight_decay}")
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-2,
        end_factor=1.0,
        total_iters=1 * len(train_loader)  # 1 epoch warmup
    )

    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(args.epochs - 1) * len(train_loader),  
        eta_min=1e-6
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[1 * len(train_loader)]  
    )
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if args.use_class_weights else None)
    contrastive_criterion = None
    if args.contrastive_method == 'supcon':
        contrastive_criterion = SupervisedContrastiveLoss(temperature=args.temperature).to(device)
    elif args.contrastive_method == 'simclr':
        contrastive_criterion = ContrastiveLoss(temperature=args.temperature).to(device)
    feature_matching_criterion = FeatureMatchingLoss(args.moving_average_decay).to(device) if args.use_feature_matching else None
    
    best_val_acc = 0
    accumulation_steps = 2  # Effective batch size = batch_size * accumulation_steps
    for epoch in range(args.epochs):
        torch.cuda.empty_cache()
        gc.collect()
        model.train()
        train_loss = 0.0
        train_cls_loss = 0.0
        train_cont_loss = 0.0
        train_feature_matching_loss = 0.0
        
        for step, (sar_img1, sar_img2, eo_img, labels) in enumerate(train_loader): 
            sar_img1 = sar_img1.to(device)
            eo_img = eo_img.to(device)
            labels = labels.to(device)
            
            # 기본 forward pass
            outputs1, sar_features1, eo_features, sar_proj1, confidence = model(sar_img1, eo_img, training=True)
            
            # 기본 classification loss
            if args.use_confidence and confidence is not None:
                # One-hot encoding
                labels_onehot = Variable(encode_onehot(labels, outputs1.size(-1)))
                
                # Confidence 처리
                confidence = torch.sigmoid(confidence)
                eps = 1e-12
                confidence = torch.clamp(confidence, 0. + eps, 1. - eps)
                
                # Random bernoulli mask
                b = Variable(torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1))).to(device)
                conf = confidence * b + (1 - b)
                
                # Combine predictions with confidence
                pred_new = outputs1 * conf.expand_as(outputs1) + \
                          labels_onehot * (1 - conf.expand_as(labels_onehot))
                
                # Classification loss
                cls_loss = criterion(pred_new, labels)
                
                # Confidence loss
                confidence_loss = -torch.log(confidence).mean()

                cls_loss = cls_loss + args.confidence_lambda * confidence_loss
                
                # Adaptive lambda 조정
                
                if 0.3 > confidence_loss.item():
                    args.confidence_lambda = args.confidence_lambda / 1.01
                elif 0.3 <= confidence_loss.item():
                    args.confidence_lambda = args.confidence_lambda / 0.99
                
                # Total loss
                
            else:
                cls_loss = criterion(outputs1, labels)
            
            total_loss = cls_loss
             
            # Feature matching loss
            feature_matching_loss = torch.tensor(0.0, device=device)
            
            feature_matching_loss = feature_matching_criterion(sar_features1, eo_features)
            total_loss += args.feature_matching_weight * feature_matching_loss
            
            # Contrastive loss
           
            sar_img2 = sar_img2.to(device)
            outputs2, sar_features2, _, sar_proj2, _ = model(sar_img2, None, training=True)
            cont_loss = contrastive_criterion(sar_proj1, sar_proj2, labels)
            total_loss +=  args.contrastive_weight * cont_loss  
            
            optimizer.zero_grad()
            total_loss = total_loss / accumulation_steps  # Scale loss
            total_loss.backward()
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
            
            train_loss += total_loss.item()
            train_cls_loss += cls_loss.item()
            train_cont_loss += cont_loss.item()
            train_feature_matching_loss += feature_matching_loss.item()

            # Step logging
            if step % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch [{epoch}/{args.epochs}], Step [{step}], "
                    f"Total Loss: {total_loss:.4f}, Cls Loss: {cls_loss:.4f}, "
                    f"SUPCON Loss: {cont_loss:.4f} (weighted: {args.contrastive_weight * cont_loss:.4f}), "
                    f"Feature Matching Loss: {feature_matching_loss:.4f} (weighted: {args.feature_matching_weight * feature_matching_loss:.4f}), "
                    f"LR: {current_lr:.6f}"
                )
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_cls_loss = train_cls_loss / len(train_loader)
        avg_train_cont_loss = train_cont_loss / len(train_loader)
        avg_train_feature_matching_loss = train_feature_matching_loss / len(train_loader)

        # Epoch logging
        writer.add_scalar('Train/Epoch_Loss', avg_train_loss, epoch)
        writer.add_scalar('Train/Epoch_cls_Loss', avg_train_cls_loss, epoch)
        if args.contrastive_method != 'none':
            writer.add_scalar('Train/Epoch_cont_Loss', avg_train_cont_loss, epoch)
            writer.add_scalar('Train/Epoch_cont_Loss_Weighted', args.contrastive_weight * avg_train_cont_loss, epoch)
        writer.add_scalar('Train/Epoch_feature_matching_Loss', avg_train_feature_matching_loss, epoch)

        log_message = (
            f'Epoch [{epoch}/{args.epochs}] - '
            f'Train Loss: {avg_train_loss:.4f}, '
            f'Cls Loss: {avg_train_cls_loss:.4f}'
        )
        
        if args.contrastive_method != 'none':
            log_message += f', {args.contrastive_method.upper()} Loss: {avg_train_cont_loss:.4f}'
            
        logger.info(log_message)
        
        if (epoch + 1) % args.val_iter == 0:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for sar_img1, sar_img2, eo_img, labels in val_loader:
                    sar_img1 = sar_img1.to(device)
                    labels = labels.to(device)

                    # inference 시에는 SAR 이미지만 사용
                    outputs1, _, _, _, confidence = model(sar_img1, training=False)
                    
                    # confidence 적용 (training과 동일한 방식)
                    
                    loss = F.cross_entropy(outputs1, labels)
                    _, preds = torch.max(outputs1, 1)
                    
                    val_loss += loss.item()
                    val_total += labels.size(0)
                    val_correct += (preds == labels).sum().item()
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            val_acc = 100 * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)

            print(f"\nEpoch {epoch+1} Summary:")
            writer.add_scalar('Val/Accuracy', val_acc, epoch)
            writer.add_scalar('Val/Loss', avg_val_loss, epoch)
            

            logger.info(
                f'Validation at Epoch [{epoch}/{args.epochs}] - '
                f'Accuracy: {val_acc:.2f}%, Loss: {avg_val_loss:.4f}'
            )
            best_path = os.path.join(exp_dir, f'{epoch}_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }, best_path)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
                checkpoint_path = os.path.join(exp_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                }, checkpoint_path)
                logger.info(f'Saved best model checkpoint to {checkpoint_path}')
            
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=train_dataset.classes,
                yticklabels=train_dataset.classes
            )
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix (Epoch {epoch+1}, Acc: {val_acc:.2f}%)')
            plt.tight_layout()
            plt.savefig(os.path.join(exp_dir, f"confusion_matrix.png"))
        
    writer.close()
    logger.info("Training completed!")

if __name__ == "__main__":
    main()