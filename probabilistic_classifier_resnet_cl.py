#This code is based on: https://github.com/SimonKohl/probabilistic_unet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=2048, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class EO_SAR_Model(nn.Module):
    def __init__(self, num_classes=10, feature_dim=2048, eo_pretrained=None, device=None, use_contrastive=False, use_confidence=False):
        super(EO_SAR_Model, self).__init__() 
        
        # Initialize encoders without pretrained weights
        self.eo_encoder = models.resnet101(weights=None)
        self.sar_encoder = models.resnet101(weights=None)
        
        # Remove classification heads first
        self.eo_encoder.fc = nn.Identity()
        self.sar_encoder.fc = nn.Identity()
        
        # Load custom pretrained weights if provided
        if eo_pretrained:
            try:
                pretrained_dict = torch.load(eo_pretrained, map_location=device)
                
                if 'model_state_dict' in pretrained_dict:
                    pretrained_dict = pretrained_dict['model_state_dict']
                
                # Filter out projection head keys and fc layer
                model_dict = self.eo_encoder.state_dict()
                pretrained_filtered_dict = {
                    k: v for k, v in pretrained_dict.items()
                    if k in model_dict and not k.startswith(('sar_proj', 'eo_proj', 'fc'))
                }
                
                # Load filtered weights
                model_dict.update(pretrained_filtered_dict)
                self.eo_encoder.load_state_dict(model_dict, strict=False)
                print(f"Successfully loaded pretrained EO model")
                
                # Freeze EO encoder
                for param in self.eo_encoder.parameters():
                    param.requires_grad = False
                
            except Exception as e:
                print(f"Warning in loading pretrained weights: {str(e)}")
        
        # Initialize classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Confidence head 추가
        self.use_confidence = use_confidence
        if use_confidence:
            self.confidence_head = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 1)
            )
        
        self.feature_dim = feature_dim
        
        self.sar_features = None
        self.eo_features = None
        
        # Projection head는 contrastive learning 사용할 때만 초기화
        self.sar_proj = ProjectionHead().to(device) if use_contrastive else None
        
    def forward(self, sar_img, eo_img=None, training=False):
        sar_features = self.sar_encoder(sar_img)
        eo_features = None
        sar_proj = None
        
        if training and eo_img is not None:
            with torch.no_grad():  
                eo_features = self.eo_encoder(eo_img)
            
            # Contrastive learning 사용할 때만 projection 수행
            if self.sar_proj is not None:
                sar_proj = self.sar_proj(sar_features)
                sar_proj = F.normalize(sar_proj, dim=1)  # L2 normalization
        
        # Classification은 SAR feature로만
        outputs = self.classifier(sar_features)
        
        # Confidence 계산 추가
        confidence = None
        if self.use_confidence:
            confidence = self.confidence_head(sar_features)
        
        return outputs, sar_features, eo_features, sar_proj, confidence
    
    def get_features(self):
       
        return self.sar_features, self.eo_features