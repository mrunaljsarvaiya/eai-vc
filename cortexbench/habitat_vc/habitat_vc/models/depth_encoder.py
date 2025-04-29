import torch
from torch import nn
from .depth_anything_v2.dpt import DepthAnythingFeatures
from torchvision.models import resnet34
from torch.nn import functional as F


################ ENCODER #####################
class Encoder(nn.Module):
    def __init__(self, n_kernels, repr_dim, dropout=0.1):
        super().__init__()
        self.n_kernels = n_kernels
        self.repr_dim = repr_dim
        self.dropout = dropout

        self.encoder = self.make_encoder(repr_dim, dropout)
    
    def make_encoder(self, repr_dim, dropout):

        return DepthEncoder(repr_dim)
    
    def forward(self, x):
        x = self.encoder(x)
        return x

    

################# DepthAnything Encoder ####################
class DepthEncoder(nn.Module):
    def __init__(self, repr_dim):
        super().__init__()

        self.config =  {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
        
        self.repr_dim = repr_dim
        self.model = DepthAnythingFeatures(**self.config['vits'])
        self.model.load_state_dict(torch.load('/scratch/ms14625/eai-vc/vc_models/src/depth_anything_v2_vits.pth', map_location='cpu'))
        self.linear = nn.Linear(384, repr_dim)

    def forward(self, x):
        x = self.model.extract_pooled_feature(x)
        x = self.linear(x)
        x = F.relu(x)

        return x
  
if __name__ == "__main__":
    # Test the encoder 

    encoder_type = 'depth_anything'  # 'resnet', 'depth_anything', 'dino_patch', 'dino_cls'
    unroll_length = 5
    repr_dim = 512

    model = Encoder(encoder_type=encoder_type, n_kernels=4, repr_dim=repr_dim, dropout=0.1)
    
    # Dummy input
    dummy_input = torch.randn(2*unroll_length, 3, 128, 128)
    
    # Forward pass
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Should be [1, 1, 518, 518]