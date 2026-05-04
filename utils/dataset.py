import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class EyeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Directory with 'awake' and 'drowsy' folders
            transform: Optional transforms
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load awake images (label 0)
        awake_dir = os.path.join(root_dir, 'awake')
        if os.path.exists(awake_dir):
            for img_name in os.listdir(awake_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(awake_dir, img_name))
                    self.labels.append(0)
        
        # Load drowsy images (label 1)
        drowsy_dir = os.path.join(root_dir, 'drowsy')
        if os.path.exists(drowsy_dir):
            for img_name in os.listdir(drowsy_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(drowsy_dir, img_name))
                    self.labels.append(1)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')  # Grayscale
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_loaders(data_dir, batch_size=32):
    """Create train and validation data loaders"""
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    dataset = EyeDataset(data_dir, transform=transform)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=2)
    
    return train_loader, val_loader
