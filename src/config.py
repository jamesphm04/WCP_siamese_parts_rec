from dataclasses import dataclass
from datetime import datetime

@dataclass
class Config:
    # Data
    train_dir: str = 'data/parts/train'
    val_dir: str = 'data/parts/val'
    image_size: int = 105
    
    #training 
    batch_size: int = 16
    num_workers: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 50
    embedding_dim: int = 256
    
    #model
    today_date = datetime.today().strftime('%Y-%m-%d')
    model_save_path: str = f'models/{today_date}_siamese.pth'
    
    #threshold for classifying pairs as similar
    threshold: float = 0.5
    
    