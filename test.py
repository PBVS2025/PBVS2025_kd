import torch
from torch.utils.data import DataLoader, Dataset
from probabilistic_classifier_resnet_cl import EO_SAR_Model
from torchvision import transforms
import os
from PIL import Image
import argparse
import csv
import zipfile  # zip 파일 생성을 위해 추가
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
from collections import defaultdict

# 클래스 인덱스 매핑
IDX2IDX = {
    0: 1,  # 'SUV' -> 'suv'
    1: 4,  # 'box_truck' -> 'box truck'
    2: 7,  # 'bus' -> 'bus'
    3: 6,  # 'flatbed_truck' -> 'flatbed truck'
    4: 5,  # 'motorcycle' -> 'motorcycle'
    5: 2,  # 'pickup_truck' -> 'pickup truck'
    6: 8,  # 'pickup_truck_w_trailer' -> 'pickup truck with trailer'
    7: 0,  # 'sedan' -> 'sedan'
    8: 9,  # 'semi_w_trailer' -> 'semi truck with trailer'
    9: 3,  # 'van' -> 'van'
}
# Challenge test 데이터셋용 클래스
class SarTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.sar_paths = []
        self.image_names = []  # 파일 이름 저장용
        
        # SAR 이미지 파일 리스트 수집
        for file_name in sorted(os.listdir(root_dir)):
            if file_name.endswith(('.jpg', '.png')):  # 적절한 파일 확장자 추가
                self.sar_paths.append(os.path.join(root_dir, file_name))
                self.image_names.append(file_name)

    def __len__(self):
        return len(self.sar_paths)

    def __getitem__(self, idx):
        sar_path = self.sar_paths[idx]
        sar_image = Image.open(sar_path).convert('RGB')
        
        if self.transform:
            sar_image = self.transform(sar_image)
            
        return sar_image, self.image_names[idx]

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='path to checkpoint')
    parser.add_argument('--test_dir', type=str, default='/home/whisper2024/PBVS/test',help='/home/whisper2024/PBVS/test')
    parser.add_argument('--output', type=str, default='submission.zip', help='output zip file path')
    args = parser.parse_args()

    # Device 설정 - 여기서 문제가 발생했습니다. device 0만 있으므로 0을 사용합니다.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Transform 설정
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4058, 0.4058, 0.4058],
            std=[0.1282, 0.1282, 0.1282]
        )
    ])

    # Dataset & DataLoader
    test_dataset = SarTestDataset(root_dir=args.test_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 모델 생성
    model = EO_SAR_Model(
        num_classes=10,
        eo_pretrained=None,  # EO pretrained 제거
        device=device
    )
    model = model.to(device)

    # Checkpoint 로드
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # projection head 관련 키 제거
        state_dict = {k: v for k, v in state_dict.items() 
                     if not any(x in k for x in ['sar_proj', 'eo_proj'])}
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Inference
    model.eval()
    network_output = np.array([])
    scores = np.array([])
    names = []
    predictions = []

    with torch.no_grad():
        for sar_img, img_name in test_loader:
            sar_img = sar_img.to(device)

            # SAR 이미지만 사용하여 inference
            outputs, _, _ , _, confidnece= model(sar_img, training=False)

            # 예측값과 confidence score 저장
            score, predicted = torch.max(outputs.data, 1)
            network_output = np.concatenate((network_output, predicted.detach().cpu().numpy()))
            scores = np.concatenate((scores, score.detach().cpu().numpy()))
            names.extend(img_name)

            if len(names) % 100 == 0:
                print(f'Processed {len(names)}/{len(test_loader)} images')

    # 결과를 CSV 파일로 저장
    csv_filename = 'results.csv'
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_id', 'class_id', 'score'])
        
        for idx, name in enumerate(names):
            image_id = name.replace('Gotcha', '').replace('.png', '')
            writer.writerow([
                image_id, 
                IDX2IDX[int(network_output[idx])],
                scores[idx]
            ])

    # ZIP 파일 생성
    with zipfile.ZipFile(args.output, 'w') as zipf:
        # CSV 파일 추가
        zipf.write(csv_filename, csv_filename)
        # readme.txt 추가
        zipf.write('readme.txt', 'readme.txt')

    # 임시 CSV 파일 삭제
    os.remove(csv_filename)

    print(f'\nSubmission zip file created at {args.output}')

if __name__ == '__main__':
    test()