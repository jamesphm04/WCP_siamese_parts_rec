import torch
import torch.nn as nn
import torch.nn.init as init

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=2048, dropout_prob=0.5):
        super(SiameseNetwork, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 10),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 4),
            nn.ReLU()
        )
        
        self.embedding = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(4096, embedding_dim),
            nn.Dropout(dropout_prob)
        )

    def forward_one(self, x: torch.Tensor):
        x = self.feature_extractor(x)
        x = x.view(x.size()[0], -1)  # Flatten the tensor
        x = self.embedding(x)
        return x
    
    def forward(self, anchor, positive, negative):
        """Forward pass for triplet input"""
        anchor_embedding = self.forward_one(anchor)
        positive_embedding = self.forward_one(positive)
        negative_embedding = self.forward_one(negative)
        return anchor_embedding, positive_embedding, negative_embedding
