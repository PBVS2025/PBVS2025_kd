import torch
import torch.nn as nn
import torch.nn.functional as F


def contrastive_loss(x, y, labels, temperature=0.1):
    """
    x: SAR features [N, D]
    y: EO features [N, D]
    labels: class labels [N]
    """
    batch_size = x.size(0)
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(x, y.T) / temperature  # [N, N]
    
    # Create mask for positive pairs (same class)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # [N, N]
    
    # For numerical stability
    sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0]
    
    # Compute log_prob
    exp_sim = torch.exp(sim_matrix)
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
    
    # Compute mean of positive pairs
    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
    
    # Loss
    loss = -mean_log_prob_pos.mean()
    
    return loss


class FeatureMatchingLoss(nn.Module):
    """Feature Matching Loss from GFMN paper"""
    def __init__(self, moving_average_decay=0.9):
        super().__init__()
        self.decay = moving_average_decay
        self.register_buffer('running_mean', None)
        self.register_buffer('running_var', None)
        
    def forward(self, sar_features, eo_features):
        # Calculate batch statistics (without L2 normalization)
        sar_mean = torch.mean(sar_features, 0)
        sar_var = torch.var(sar_features, 0)
        eo_mean = torch.mean(eo_features, 0)
        eo_var = torch.var(eo_features, 0)
        
        # Update running statistics
        if self.running_mean is None:
            self.running_mean = eo_mean.detach()
            self.running_var = eo_var.detach()
        else:
            self.running_mean = self.decay * self.running_mean + (1 - self.decay) * eo_mean.detach()
            self.running_var = self.decay * self.running_var + (1 - self.decay) * eo_var.detach()
            
        # Feature matching loss with MSE and relative error
        mean_loss = F.mse_loss(sar_mean, self.running_mean)
        var_loss = F.mse_loss(sar_var, self.running_var)
        
        # Add relative error terms
        relative_mean_loss = torch.mean(torch.abs(sar_mean - self.running_mean) / (torch.abs(self.running_mean) + 1e-6))
        relative_var_loss = torch.mean(torch.abs(sar_var - self.running_var) / (torch.abs(self.running_var) + 1e-6))
        
        return mean_loss + var_loss + relative_mean_loss + relative_var_loss

def unsupervised_contrastive_loss(x, y, temperature=0.5):
    batch_size = x.size(0)
    
    # L2 normalization 이미 명시적으로 지정됨
    x = F.normalize(x, dim=1, p=2)
    y = F.normalize(y, dim=1, p=2)
    
    # Positive pair의 similarity
    pos_sim = torch.sum(x * y, dim=1) / temperature
    
    # All pairs의 similarity
    all_sim = torch.matmul(x, y.T) / temperature
    
    # Negative pair만 선택 (자기 자신 제외)
    mask = torch.eye(batch_size, device=x.device)
    neg_sim = all_sim * (1 - mask)
    
    # InfoNCE loss
    numerator = torch.exp(pos_sim)
    denominator = numerator + torch.sum(torch.exp(neg_sim), dim=1)
    
    loss = -torch.log(numerator / denominator).mean()
    
    return loss

def supervised_contrastive_loss(x, y, labels, temperature=5.0, k=5):
    batch_size = x.size(0)
    
    x = F.normalize(x, dim=1, p=2)
    
    # y가 None이면 x를 y로 사용 (같은 feature의 두 augmentation view 간 대조학습)
    if y is None:
        y = x
    else:
        y = F.normalize(y, dim=1, p=2)
    
    # Compute similarity matrix between two views
    sim_matrix = torch.matmul(x, y.T) / temperature  # [N, N]
    
    # Positive mask (같은 클래스의 샘플들)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # [N, N]
    
    # Hard negative mining with class balancing
    neg_mask = 1 - pos_mask
    sim_matrix_neg = sim_matrix * neg_mask
    
    # Class-wise normalization for balanced negative sampling
    label_counts = torch.bincount(labels)
    class_weights = 1.0 / label_counts[labels].float()
    class_weights = class_weights.unsqueeze(0)  # [1, B]
    
    # Apply class weights to negative similarities
    sim_matrix_neg = sim_matrix_neg * class_weights
    
    # Select k hard negatives per anchor
    k = min(k, batch_size - 1)  # k는 배치 크기보다 작아야 함
    _, hard_indices = sim_matrix_neg.topk(k, dim=1)
    hard_neg_mask = torch.zeros_like(neg_mask, device=x.device)
    hard_neg_mask.scatter_(1, hard_indices, 1.0)
    
    # Compute loss with balanced negatives
    sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0]  # numerical stability
    exp_sim = torch.exp(sim_matrix)
    
    pos_sim = exp_sim * pos_mask
    neg_sim = exp_sim * hard_neg_mask
    
    loss = -torch.log(pos_sim.sum(dim=1) / (pos_sim.sum(dim=1) + neg_sim.sum(dim=1) + 1e-8))
    
    valid_mask = (pos_mask.sum(dim=1) > 0)
    loss = (loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
    
    return loss

class ContrastiveLoss(nn.Module):
    """Unsupervised Contrastive Loss"""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, x, y, labels=None):
        return unsupervised_contrastive_loss(x, y, self.temperature)

class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss"""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, x, y, labels):
        return supervised_contrastive_loss(x, y, labels, self.temperature)