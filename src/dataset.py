import os 
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple
import random

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
        
        # Get all image paths and their labels
        self.images = []
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.images.append((img_path, class_name))
                
    def len(self):
        return len(self.images)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img1_path, class1 = self.images[idx]
        
        # randomly decide if we want a matching pair (1) or non-matching pair (0)
        should_match = random.random() > 0.5
        
        if should_match:
            # get another image from the same class
            possible_matches = [(p, c) for p, c in self.images if c == class1 and p != img1_path]
            img2_path, class2 = random.choice(possible_matches)
            target = 1.0
        else:
            # get another image from a different class
            possible_matches = [(p, c) for p, c in self.images if c != class1]
            img2_path, class2 = random.choice(possible_matches)
            target = 0.0
            
        # load and transform images 
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2, torch.tensor(target, dtype=torch.float32), torch.tensor([self.class_to_idx[class1], self.class_to_idx[class2]])
    
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
        