from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from config import Config

path = os.path.dirname(__file__)+"/../data/1.jpg"
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

class ImageDataset(Dataset):
    def __init__(self, path, config: Config):
        self.path = path
        self.config = config
        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std)
        ])
        image = Image.open(self.path)
        self.image = self.transform(image)

    def normalize(self, image):
        for i in range(3):
            image[i] = image[i]*std[i]+mean[i]
        return image
    
    def show(self, image = None):
        if image is None:
            image = self.image
        image = self.normalize(image)
        image = image.permute(1,2,0)
        plt.imshow(image)
        plt.show()

    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.image.to(self.config.device)

if __name__ == "__main__":
    dataset = ImageDataset(path)
    dataset.show()


