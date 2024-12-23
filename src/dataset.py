import os 
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple
import random
import torch.nn as nn

class AutoPartsDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose(
            [
            transforms.Resize((105, 105)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        
        # Get all classes (part types)
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {part: idx for idx, part in enumerate(self.classes)}
        
        # Create class-to-images mapping
        self.class_to_images = {}
        self.classes = []
        
        # Scan directory
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                self.classes.append(class_name)
                self.class_to_images[class_name] = [
                    os.path.join(class_path, img_name)
                    for img_name in os.listdir(class_path)
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
        
        # Create triplets
        self.triplets = self._generate_triplets()
                
    def __len__(self):
        return len(self.triplets)

    def _generate_triplets(self):
        triplets = []
        
        for anchor_class in self.classes:
            anchor_images = self.class_to_images[anchor_class]
            
            for anchor_img in anchor_images:
                # get positive image from the same class
                positive_images = [img for img in anchor_images if img != anchor_img]
                if not positive_images: 
                    continue
                positive_img = random.choice(positive_images)
                
                # get negative image from a different class
                
                negative_class = random.choice([c for c in self.classes if c != anchor_class])
                negative_img = random.choice(self.class_to_images[negative_class])
                
                triplets.append((anchor_img, positive_img, negative_img))
                
        return triplets
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        anchor_img_path, positive_img_path, negative_img_path = self.triplets[idx]
        
        # load and transform images 
        anchor_img = Image.open(anchor_img_path).convert('RGB')
        positive_img = Image.open(positive_img_path).convert('RGB')
        negative_img = Image.open(negative_img_path).convert('RGB')
        
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
            
        return anchor_img, positive_img, negative_img
    
def get_data_loaders(train_dir: str, val_dir: str, batch_size: int = 32, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    train_dataset = AutoPartsDataset(train_dir)
    val_dataset = AutoPartsDataset(val_dir)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = torch.nn.functional.pairwise_distance(anchor, positive, p=2)
        distance_negative = torch.nn.functional.pairwise_distance(anchor, negative, p=2)
        
        losses = torch.nn.functional.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()