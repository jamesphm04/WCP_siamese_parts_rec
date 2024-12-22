import torch 
import torch.nn as nn 
from torchvision.models import resnet50, ResNet50_Weights

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=2048):
        super(SiameseNetwork, self).__init__()
        
        # load the pretrain model 
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # remove the last fully connected layer 
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # add embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(2048, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # similarity head
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward_one(self, x: torch.Tensor):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return x
    
    def forward(self, x1, x2):
        x1 = self.forward_one(x1)
        x2 = self.forward_one(x2)
        
        # compute similarity score
        x = torch.abs(x1 - x2)
        out = self.fc(x)
        return out