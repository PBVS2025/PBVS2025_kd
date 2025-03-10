import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)

def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

def save_mask_prediction_example(mask, pred, iter):
	plt.imshow(pred[0,:,:],cmap='Greys')
	plt.savefig('images/'+str(iter)+"_prediction.png")
	plt.imshow(mask[0,:,:],cmap='Greys')
	plt.savefig('images/'+str(iter)+"_mask.png")


def compute_adjustment(train_loader, tro, device):
    """compute the base probabilities with progress tracking"""
    print("클래스 빈도 계산 시작...")
    
    # 전체 데이터 개수 미리 파악 (진행률 계산용)
    total_samples = len(train_loader.dataset)
    processed_samples = 0
    
    label_freq = {}
    for i, (inputs, _, target) in enumerate(train_loader):
        batch_size = target.size(0)
        processed_samples += batch_size
        progress = 100 * processed_samples / total_samples
        
        # 진행률 출력 (10%마다 또는 일정 배치마다)
        if i % 10 == 0 or processed_samples == total_samples:
            print(f"진행률: {progress:.1f}% ({processed_samples}/{total_samples})")
            
        target = target.to(device)
        for j in target:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
            
    print("클래스별 빈도 계산 완료!")
    
    # 결과 계산
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    
    print("로짓 조정값 계산 중...")
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(device)
    
    print("조정값 계산 완료!")
    print(f"클래스 빈도: {label_freq_array}")
    print(f"로짓 조정값: {adjustments}")
    
    return adjustments