import os
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class GrayTo3Channels(object):
    def __call__(self, img):
        return img.convert('RGB')

weak_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    GrayTo3Channels(),
    transforms.ToTensor(),
])

strong_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.RandomRotation(15)], p=0.5),
    transforms.RandomResizedCrop(size=64, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    GrayTo3Channels(),
    transforms.ToTensor(),
])

class LabeledImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.df['Class_encoded'] = self.label_encoder.fit_transform(self.df['Class'])
        self.df = self.df[self.df['image_path'].apply(os.path.exists)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'image_path']
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.df.loc[idx, 'Class_encoded']).long()
        return image, label

class UnlabeledImageDataset(Dataset):
    def __init__(self, df, weak_transform=None, strong_transform=None):
        self.df = df.reset_index(drop=True)
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.df = self.df[self.df['image_path'].apply(os.path.exists)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'image_path']
        image = Image.open(img_path).convert('L')
        weak_image = self.weak_transform(image)
        strong_image = self.strong_transform(image)
        return weak_image, strong_image
