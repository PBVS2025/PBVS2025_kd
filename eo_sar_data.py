import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import random
from PIL import Image
from torchvision import transforms

class SarEODataset(Dataset):
    def __init__(self, root_dir, sar_transform=None, eo_transform=None, use_contrastive=False):
        self.root_dir = root_dir
        self.sar_transform = sar_transform
        self.eo_transform = eo_transform
        self.use_contrastive = use_contrastive  # contrastive learning 사용 여부
        
        print(f"Initializing dataset from {root_dir}")  # 디버깅용 로그
        
        self.sar_root = os.path.join(root_dir, 'SAR_Train')
        self.eo_root = os.path.join(root_dir, 'EO_Train')
        
        # 디렉토리 존재 확인
        if not os.path.exists(self.sar_root):
            raise ValueError(f"SAR directory not found: {self.sar_root}")
        if not os.path.exists(self.eo_root):
            raise ValueError(f"EO directory not found: {self.eo_root}")
            
        print(f"Found SAR directory: {self.sar_root}")  # 디버깅용 로그
        print(f"Found EO directory: {self.eo_root}")    # 디버깅용 로그
        
        self.classes = sorted(os.listdir(self.sar_root))
        print(f"Found classes: {self.classes}")  # 디버깅용 로그
        
        self.cls_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.sar_paths = []
        self.eo_paths = []
        self.labels = []
        
        self.class_counts = [0] * len(self.classes)  # 클래스별 카운트 저장
        
        total_samples = 0
        for class_name in self.classes:
            sar_class_dir = os.path.join(self.sar_root, class_name)
            eo_class_dir = os.path.join(self.eo_root, class_name)
            
            print(f"Processing class {class_name}")  # 디버깅용 로그

            sar_files = sorted([f for f in os.listdir(sar_class_dir) 
                              if f.lower().endswith('.png')])
            
            class_samples = 0
            for sar_file in sar_files:
                eo_file = sar_file
                eo_file_path = os.path.join(eo_class_dir, eo_file)
                
                if os.path.exists(eo_file_path):
                    self.sar_paths.append(os.path.join(sar_class_dir, sar_file))
                    self.eo_paths.append(eo_file_path)
                    self.labels.append(self.cls_to_idx[class_name])
                    class_samples += 1
            
            self.class_counts[self.cls_to_idx[class_name]] = class_samples  # 클래스별 카운트 저장
            total_samples += class_samples
            print(f"Added {class_samples} samples for class {class_name}")  # 디버깅용 로그
            
        print(f"Total samples loaded: {total_samples}")  # 디버깅용 로그

    def __len__(self):
        return len(self.sar_paths)

    def __getitem__(self, idx):
        sar_path = self.sar_paths[idx]
        eo_path = self.eo_paths[idx]
        
        sar_img = Image.open(sar_path).convert('RGB')
        eo_img = Image.open(eo_path).convert('RGB')
        
        # 기본 SAR view
        sar_img1 = self.sar_transform(sar_img)
        
        # Contrastive learning 사용할 때만 두 번째 view 생성
        if self.use_contrastive:
            sar_img2 = self.sar_transform(sar_img)  # 다른 random augmentation
        else:
            # None 대신 빈 텐서 반환
            sar_img2 = torch.zeros_like(sar_img1)  # 같은 shape의 0으로 채워진 텐서
        
        if self.eo_transform:
            eo_img = self.eo_transform(eo_img)
        
        label = self.labels[idx]
        
        return sar_img1, sar_img2, eo_img, label

    def get_class_counts(self):
        """Return the number of samples for each class"""
        return self.class_counts

    def get_labels(self):
        """Returns a list of labels for ImbalancedDatasetSampler"""
        return self.labels