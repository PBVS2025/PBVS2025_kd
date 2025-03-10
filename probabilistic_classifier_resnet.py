#This code is based on: https://github.com/SimonKohl/probabilistic_unet

from unet_blocks import *
from utils import init_weights,init_weights_orthogonal_normal, l2_regularisation
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
import torch.nn.functional as F
import torchvision.models as models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, initializers, posterior=False):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'
        self.encoder =  models.resnet50(weights='IMAGENET1K_V2')
        self.features = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu,
            self.encoder.maxpool,
            self.encoder.layer1,
            self.encoder.layer2,
            self.encoder.layer3,
            self.encoder.layer4
        )
        
        self.conv_layer = nn.Conv2d(2048, 2 * self.latent_dim, (1,1), stride=1)
        
        
    

    def forward(self, input, segm=None):

       
        if segm is not None:
            input = torch.cat((input, segm), dim=1)

        encoding = self.features(input)  # [batch_size, 2048, H, W]
        encoding = torch.mean(encoding, dim=[2,3], keepdim=True)
        
        mu_log_sigma = self.conv_layer(encoding)
        mu_log_sigma = mu_log_sigma.squeeze(-1).squeeze(-1)

        mu = mu_log_sigma[:,:self.latent_dim]
        log_sigma = mu_log_sigma[:,self.latent_dim:]

        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)
        return dist


class ProbabilisticClassifier(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, num_filters=[32,64,128,192], 
                 latent_dim=6, no_convs_fcomb=4, beta=10.0, use_adjustment = False):
                #  use_confidence= False, confidence_lambda = 1.0):
        super(ProbabilisticClassifier, self).__init__()


        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.beta = beta
        self.use_adjustments = use_adjustment
        self.feature_extractor = models.resnet101(weights='IMAGENET1K_V2')
        self.feature_layer = nn.Sequential(
            self.feature_extractor.conv1,
            self.feature_extractor.bn1,
            self.feature_extractor.relu,
            self.feature_extractor.maxpool,
            self.feature_extractor.layer1,
            self.feature_extractor.layer2,
            self.feature_extractor.layer3,
            self.feature_extractor.layer4
        )
        
        self.prior = AxisAlignedConvGaussian(
            input_channels=input_channels,
            num_filters=num_filters,
            no_convs_per_block=self.no_convs_per_block,
            latent_dim=latent_dim,
            initializers={'w':'he_normal', 'b':'normal'}
        )

        
        self.posterior = AxisAlignedConvGaussian(
            input_channels=input_channels,
            num_filters=num_filters,
            no_convs_per_block=self.no_convs_per_block,
            latent_dim=latent_dim,
            initializers={'w':'he_normal', 'b':'normal'},
            posterior=True
        )

       
        self.classifier = nn.Sequential(
            nn.Linear(2048 + latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, sar_img, eo_img=None, training=True):
        if training and eo_img is not None:
            # EO로부터 posterior 분포 얻기
            self.posterior_latent_space = self.posterior(eo_img)

        # SAR로부터 prior 분포 얻기
        self.prior_latent_space = self.prior(sar_img)
        
        # SAR feature extraction
        self.features = self.feature_layer(sar_img)
        # Global average pooling
        self.features = torch.mean(self.features, dim=[2,3])

    def sample(self, testing=False):
        if testing:
            z_prior = self.prior_latent_space.sample()
        else:
            z_prior = self.prior_latent_space.rsample()
        
        # Combine features with latent sample
        combined = torch.cat([self.features, z_prior], dim=1)
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

    def elbo(self, labels, analytic_kl=True, adjustments=None):
        criterion = nn.CrossEntropyLoss(reduction='mean')
        
        self.logits = self.sample(testing=False)
        

        if hasattr(self, 'use_adjustments') and self.use_adjustments and adjustments is not None:
            
            if not hasattr(self, '_logged_adjustment') or self._logged_adjustment < 5:
                if not hasattr(self, '_logged_adjustment'):
                    self._logged_adjustment = 0
                self._logged_adjustment += 1
                
           
            adjusted_logits = self.logits + adjustments
            
            self.cls_loss = F.cross_entropy(adjusted_logits, labels)
        else:
           
            if not hasattr(self, '_logged_no_adjustment') or self._logged_no_adjustment < 1:
                if not hasattr(self, '_logged_no_adjustment'):
                    self._logged_no_adjustment = 0
                self._logged_no_adjustment += 1
                
                print("--- No Adjustment Applied ---")
                if not hasattr(self, 'use_adjustments'):
                    print("use_adjustments flag is not set")
                elif not self.use_adjustments:
                    print("use_adjustments is set to False")
                elif adjustments is None:
                    print("adjustments is None")
                print("Using original logits for loss calculation")
                print("-----------------------------------")
            
            # 기존 방식대로 손실 계산
            self.cls_loss = F.cross_entropy(self.logits, labels)
            
        # Loss 계산
        self.kl = torch.mean(self.kl_divergence(analytic=analytic_kl))
        
        return -(self.cls_loss + self.beta * self.kl)