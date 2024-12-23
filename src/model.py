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
        x = nn.functional.normalize(x, p=2, dim=1)
        return x
    
    def forward(self, anchor, positive, negative):
        """Forward pass for triplet input"""
        anchor_emb = self.forward_one(anchor)
        positive_emb = self.forward_one(positive)
        negative_emb = self.forward_one(negative)
        return anchor_emb, positive_emb, negative_emb