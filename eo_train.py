import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
import os
from tqdm import tqdm
from PIL import Image
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# EO 데이터셋 클래스 정의
class EODataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                        self.samples.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

# 학습 함수
def train_eo_model(model, train_loader, val_loader, optimizer, criterion, device, start_epoch=0, num_epochs=10, save_path='eo_model.pth', best_acc=0.0):
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # 학습 단계
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{start_epoch + num_epochs}') as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': running_loss / (pbar.n + 1),
                    'acc': 100. * correct / total
                })
        
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f'Epoch {epoch+1}: Validation Acc: {val_acc:.2f}%, Loss: {val_loss/len(val_loader):.4f}')
        torch.save({
            'epoch': epoch + 1,  
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': best_acc,
        }, f'{epoch}.pth')

        # 최고 성능 모델 저장
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch + 1,  
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
            }, save_path)
            print(f'Saved best model with accuracy: {best_acc:.2f}%')
    
    return model, best_acc

def main():
    parser = argparse.ArgumentParser(description='Train EO classification model')
    parser.add_argument('--resume', type=str, default='', help='path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--save_path', type=str, default='eo_resnet101.pth', help='path to save model')
    
    args = parser.parse_args()
    
    # 데이터 변환
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.2892, 0.2892, 0.2892],
            std=[0.1455, 0.1455, 0.1455]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.2892, 0.2892, 0.2892],
            std=[0.1455, 0.1455, 0.1455]
        )
    ])
    
    # 데이터셋 및 데이터로더
    train_dataset = EODataset(root_dir='/home/whisper2024/PBVS/Unicorn_Dataset/EO_Train', transform=train_transform)
    val_dataset = EODataset(root_dir='/home/whisper2024/PBVS/CL_Test/val/EO_Train', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 디렉토리 이름 순서
    dir_names = ["box_truck", "bus", "flatbed_truck", "motorcycle", "pickup_truck", 
                "pickup_truck_w_trailer", "sedan", "semi_w_trailer", "SUV", "van"]

    # 테이블에서의 클래스 이름과 샘플 수
    class_names = ["sedan", "SUV", "pickup truck", "van", "box truck", 
                "motorcycle", "flatbed truck", "bus", "pickup truck w/ trailer", 
                "semi truck w/ trailer"]
    class_counts = [364291, 43401, 24158, 16890, 2896, 1441, 898, 612, 695, 353]

    # 디렉토리 이름과 클래스 이름을 매핑하여 샘플 수 재정렬
    dir_to_class = {
        "box_truck": "box truck",
        "bus": "bus",
        "flatbed_truck": "flatbed truck",
        "motorcycle": "motorcycle",
        "pickup_truck": "pickup truck",
        "pickup_truck_w_trailer": "pickup truck w/ trailer",
        "sedan": "sedan",
        "semi_w_trailer": "semi truck w/ trailer",
        "SUV": "SUV",
        "van": "van"
    }

    # 순서에 맞게 샘플 수 재정렬
    ordered_counts = []
    for dir_name in dir_names:
        class_name = dir_to_class[dir_name]
        idx = class_names.index(class_name)
        ordered_counts.append(class_counts[idx])

    print("디렉토리 순서에 따른 클래스별 샘플 수:")
    for i, (dir_name, count) in enumerate(zip(dir_names, ordered_counts)):
        print(f"{i}: {dir_name} - {count}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 클래스 가중치 계산
    class_weights = 1. / torch.tensor(ordered_counts, dtype=torch.float)
    normalized_weights = class_weights / class_weights.sum() * len(ordered_counts)
    normalized_weights = normalized_weights.to(device)

    print("\n계산된 클래스 가중치:")
    for i, (dir_name, weight) in enumerate(zip(dir_names, normalized_weights)):
        print(f"{i}: {dir_name} - {weight:.4f}")

    # 모델 초기화
    num_classes = len(train_dataset.classes)
    start_epoch = 0
    best_acc = 0.0
    
    if args.resume:
        print(f"=> 체크포인트에서 모델 불러오는 중: '{args.resume}'")
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            
            # ResNet101 모델 생성
            model = models.resnet101()
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
            # 체크포인트에서 상태 불러오기
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            
            # 학습 재개 정보 설정
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['accuracy']
            print(f"=> 불러온 체크포인트 (epoch {start_epoch}), 최고 정확도: {best_acc:.2f}%")
        else:
            print(f"=> 체크포인트 파일을 찾을 수 없습니다: '{args.resume}'")
            return
    else:
        print("=> 새 모델로 시작합니다.")
        model = models.resnet101(weights='IMAGENET1K_V2')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)

    # 손실 함수와 옵티마이저 설정
    criterion = nn.CrossEntropyLoss(weight=normalized_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.05, nesterov=True)
    
    # 체크포인트에서 옵티마이저 상태 불러오기
    if args.resume and os.path.isfile(args.resume):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 학습률 변경 가능 (선택 사항)
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
    
    # 학습 실행
    trained_model, final_acc = train_eo_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        start_epoch=start_epoch,
        num_epochs=args.epochs,
        save_path=args.save_path,
        best_acc=best_acc
    )
    
    print(f"Training completed! 최종 최고 정확도: {final_acc:.2f}%")

if __name__ == "__main__":
    main()