import torch
from torch.utils.data import DataLoader, Dataset
from probabilistic_classifier import ProbabilisticClassifier
from torchvision import transforms
import os
from PIL import Image
import argparse
import csv
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
from collections import defaultdict
import torch
from eo_sar_data import SarEODataset
def validate_with_gt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='path to checkpoint')
    parser.add_argument('--test_dir', type=str, required=True, help = 'Dir path')
    
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    test_dataset = SarEODataset(root_dir = args.test_dir, transform = test_transform)

    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False,num_workers=4, pin_memory=True)
   
    net = ProbabilisticClassifier(
    input_channels=3,       
    num_classes=10,          
    num_filters=[32,64,128,192], 
    latent_dim=6,           
    beta=10.0
    ).to(device)
    

    checkpoint = torch.load(args.checkpoint, map_location= device)

    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    with torch.no_grad():
        for sar_img, _, labels in test_dataloader:
            sar_img = sar_img.to(device)
            labels = labels.to(device)
            
            net.forward(sar_img, training=False)
            outputs = net.sample(testing=True)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 클래스별 정확도 계산
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1
    
    print(f'\nOverall Accuracy: {100 * correct / total:.2f}%')
    print('\nClass-wise Accuracies:')
    for class_id in range(len(test_dataset.classes)):
        if class_total[class_id] > 0:
            acc = 100 * class_correct[class_id] / class_total[class_id]
            print(f'{test_dataset.classes[class_id]}: {acc:.2f}% ({class_correct[class_id]}/{class_total[class_id]})')

if __name__ == '__main__':
    validate_with_gt()



