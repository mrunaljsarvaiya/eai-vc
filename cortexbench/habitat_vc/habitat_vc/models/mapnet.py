import torch.nn as nn
import torch


class MapNet(nn.Module):
    def __init__(self, encoding_dim=256):
        super().__init__()
        
        # CNN to encode top-down maps
        self.map_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Fully connected layer to flatten map features
        self.map_fc = nn.Linear(128 * 8 * 8, encoding_dim)  # Adjust based on CNN output size

    
    def forward(self, x):
        # Encode top-down map
        x = self.map_encoder(x)
        x = x.view(x.size(0), -1)
        x = self.map_fc(x)
        return x
    
if __name__ == "__main__":
    model = MapNet()
    x = torch.randn(1, 1, 64, 64)  # Batch size 1, 1 channel, 64x64 map

    print(model(x).shape)
    # Output: torch.Size([1, 256])
    
