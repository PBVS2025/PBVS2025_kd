#This code is based on: https://github.com/SimonKohl/probabilistic_unet

from unet_blocks import *
from unet import Unet
from utils import init_weights,init_weights_orthogonal_normal, l2_regularisation
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
import torch.nn.functional as F
import torchvision.models as models
from models import vit_image as vit_models
from easydict import EasyDict
import os
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """
    def __init__(self, input_channels, num_filters, latent_dim, initializers, posterior=False):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.posterior = posterior
        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'
        # AdapterFormer 설정
        tuning_config = EasyDict(
            ffn_adapt=True,
            ffn_option="parallel",
            ffn_adapter_layernorm_option="none",
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1",
            ffn_num=64,
            d_model=768,
            vpt_on=False,
            vpt_num=0,
        )

        
        self.encoder = vit_models.vit_base_patch16(
            in_chans=self.input_channels, 
            num_classes=2 * latent_dim,  # mu와 log_sigma 출력
            global_pool=True,
            tuning_config=tuning_config
        )
        
    def forward(self, input, segm=None):

       
        if segm is not None:
            input = torch.cat((input, segm), dim=1)

        mu_log_sigma = self.encoder(input)
    
        mu = mu_log_sigma[:, :self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim:]
        
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
        return dist

class Fcomb(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """
    def __init__(self, num_filters, latent_dim, num_output_channels, num_classes, no_convs_fcomb, initializers, use_tile=True):
        super(Fcomb, self).__init__()
        self.num_channels = num_output_channels #output channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2,3]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb 
        self.name = 'Fcomb'

        if self.use_tile:
            layers = []

            #Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
            layers.append(nn.Conv2d(self.num_filters[0]+self.latent_dim, self.num_filters[0], kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_fcomb-2):
                layers.append(nn.Conv2d(self.num_filters[0], self.num_filters[0], kernel_size=1))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)

            self.last_layer = nn.Conv2d(self.num_filters[0], self.num_classes, kernel_size=1)

            if initializers['w'] == 'orthogonal':
                self.layers.apply(init_weights_orthogonal_normal)
                self.last_layer.apply(init_weights_orthogonal_normal)
            else:
                self.layers.apply(init_weights)
                self.last_layer.apply(init_weights)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z):
        """
        Z is batch_sizexlatent_dim and feature_map is batch_sizexno_channelsxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """
        if self.use_tile:
            z = torch.unsqueeze(z,2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z,3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])

            #Concatenate the feature map (output of the UNet) and the sample taken from the latent space
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
            output = self.layers(feature_map)
            return self.last_layer(output)


class ProbabilisticClassifier(nn.Module):

    def __init__(self, input_channels=3, num_classes=10, num_filters=[32,64,128,192], 
                 latent_dim=6, beta=10.0,
                 use_confidence= False, confidence_lambda = 1.0, pretrained_path=None):
        super(ProbabilisticClassifier, self).__init__()


        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.beta = beta
        self.confidence_lambda = confidence_lambda
        self.use_confidence = use_confidence
        
        tuning_config = EasyDict(
            ffn_adapt=True,
            ffn_option="parallel",
            ffn_adapter_layernorm_option="none",
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1",
            ffn_num=64,
            d_model=768,
            vpt_on=False,
            vpt_num=0,
        )
        if use_confidence:
            self.confidence_haed = nn.Linear(768 + latent_dim,1)

        self.feature_extractor = vit_models.vit_base_patch16(
            in_chans=input_channels,
            num_classes=0,  
            global_pool=True,
            tuning_config=tuning_config
        )
       
        self.prior = AxisAlignedConvGaussian(
            input_channels=input_channels,
            num_filters=num_filters,
            latent_dim=latent_dim,
            initializers={'w':'he_normal', 'b':'normal'}
        )

        
        self.posterior = AxisAlignedConvGaussian(
            input_channels=input_channels,
            num_filters=num_filters,
            latent_dim=latent_dim,
            initializers={'w':'he_normal', 'b':'normal'},
            posterior=True
        )

        if pretrained_path is not None and os.path.exists(pretrained_path):
            try:
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                model_weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
                
                # 헤드 레이어 제외
                for k in ['head.weight', 'head.bias']:
                    if k in model_weights:
                        del model_weights[k]

                
                msg = self.feature_extractor.load_state_dict(model_weights, strict=False)
                for name, p in self.feature_extractor.named_parameters():
                    if name in msg.missing_keys:  # adapter parameters
                        p.requires_grad = True
                    else:  # backbone parameters
                        p.requires_grad = False
                
                
                prior_msg = self.prior.encoder.load_state_dict(model_weights, strict=False)

                for name, p in self.prior.encoder.named_parameters():
                    if name in prior_msg.missing_keys:  # adapter parameters
                        p.requires_grad = True
                    else:  # backbone parameters
                        p.requires_grad = False

                posterior_msg =self.posterior.encoder.load_state_dict(model_weights, strict=False)
                for name, p in self.posterior.encoder.named_parameters():
                    if name in posterior_msg.missing_keys:  # adapter parameters
                        p.requires_grad = True
                    else:  # backbone parameters
                        p.requires_grad = False
            
            except Exception as e:
                print(f"Error loading pretrained weights: {e}")

       
        self.classifier = nn.Linear(768 + latent_dim, num_classes)
        

    def forward(self, sar_img, eo_img=None, training=True):
        if training and eo_img is not None:
            # EO로부터 posterior 분포 얻기
            self.posterior_latent_space = self.posterior(eo_img)

        # SAR로부터 prior 분포 얻기
        self.prior_latent_space = self.prior(sar_img)
        
        # SAR feature extraction
        self.features = self.feature_extractor(sar_img)
    

    def sample(self, testing=False):
        if testing:
            z_prior = self.prior_latent_space.sample()
        else:
            z_prior = self.prior_latent_space.rsample()
        
        # 여기서 cross attention 포함 할 것
        combined = torch.cat([self.features, z_prior], dim=1) 
        
        if self.use_confidence:
            logits = self.classifier(combined)
            confidence = self.confidence_haed(combined)
            
            return logits, confidence
        else:
            return self.classifier(combined)

    def kl_divergence(self, analytic=True):
        if analytic:
            kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        else:
            z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

    def elbo(self, labels, analytic_kl=True, use_la_loss=False, adjustments=None):
        criterion = nn.CrossEntropyLoss(reduction='mean')
        
        if self.use_confidence:
            self.logits, confidence = self.sample(testing=False)
            
            # Confidence 관련 계산
            confidence = torch.sigmoid(confidence)
            eps = 1e-12
            confidence = torch.clamp(confidence, 0. + eps, 1. - eps)
            labels_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
            
            if use_la_loss and adjustments is not None:
                # LA loss를 위한 로짓 조정
                adjusted_logits = self.logits + adjustments
                pred_original = F.softmax(adjusted_logits, dim=-1)
            else:
                pred_original = F.softmax(self.logits, dim=-1)
            
            pred_original = torch.clamp(pred_original, 0. + eps, 1. - eps)
            
            # confidence 적용
            b = torch.bernoulli(torch.ones_like(confidence) * 0.5).to(self.logits.device)
            conf = confidence * b + (1 - b)
            pred_new = pred_original * conf + labels_onehot * (1 - conf)
            pred_new = torch.log(pred_new)
            
            self.cls_loss = F.nll_loss(pred_new, labels)
            confidence_loss = -torch.log(confidence).mean()
            self.cls_loss = self.cls_loss + (self.confidence_lambda * confidence_loss)
        
        else:
            self.logits = self.sample(testing=False)
            if use_la_loss and adjustments is not None:
                adjusted_logits = self.logits + adjustments
                self.cls_loss = F.cross_entropy(adjusted_logits, labels)
            else:
                self.cls_loss = F.cross_entropy(self.logits, labels)
        
        self.kl = torch.mean(self.kl_divergence(analytic=analytic_kl))
        
        return -(self.cls_loss + self.beta * self.kl)