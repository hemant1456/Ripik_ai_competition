from torchvision import transforms
import numpy as np
import os
from PIL import Image
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from albumentations import Compose,PadIfNeeded,RandomCrop,HorizontalFlip,Normalize, Resize
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.dropout import CoarseDropout
import numpy as np


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def get_mean_std(folder_path):
    pixel_sum = np.array([0.0, 0.0, 0.0])
    pixel_square_sum = np.array([0.0, 0.0, 0.0])
    num_images = 0

    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)

        pixel_sum += img_tensor.mean([1, 2]).numpy()
        pixel_square_sum += (img_tensor**2).mean([1, 2]).numpy()
        num_images += 1

    mean = pixel_sum / num_images
    std = (pixel_square_sum / num_images - mean**2)**0.5

    return mean, std

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, str(self.dataframe.iloc[idx, 1]))  # Convert to string
        image = Image.open(img_name)
        label = self.dataframe.iloc[idx, 2]

        if self.transform:
            image = self.transform(image)

        return image, label

class TestImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_name).convert('RGB')  # Convert to 'RGB' if your images are not in this mode

        if self.transform:
            image = self.transform(image)

        return image, self.img_names[idx]  # Return image and its name


# Applying some transformation to data

labels_df = pd.read_csv('train/train.csv')
labels_df['label'] = labels_df['label'].apply(lambda x: x - 1)

class Albumentation:
  def __init__(self,transforms):
    self.transforms= Compose(transforms)
  def __call__(self,image):
    return self.transforms(image=np.array(image))['image']

train_transforms= Albumentation([
    Resize(224,224),
    PadIfNeeded(260,260),
    RandomCrop(224,224,p=1),
    HorizontalFlip(0.5),
    CoarseDropout(min_height=50,max_height=50,min_width=50,max_width=50,max_holes=2,fill_value=(0.47768187*255, 0.45968917*255, 0.46049647*255),p=0.7),
    Normalize((0.47768187,0.45968917,0.46049647), (0.271, 0.266,0.268)),  # Normalizing
    ToTensorV2()
])

test_transforms= Albumentation([
    Resize(224,224),
    Normalize((0.47768187,0.45968917,0.46049647), (0.271, 0.266,0.268)),  # Normalizing
    ToTensorV2()
])

def get_data_loaders(batch_size=64):
    # Create datasets
    train_dataset = CustomImageDataset(dataframe=labels_df, img_dir='train/images/', transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TestImageDataset(img_dir='test/images/', transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader



def denormalise(image):
  all_means= torch.tensor([0.47768187,0.45968917,0.46049647])
  all_stds= torch.tensor([0.271, 0.266,0.268])
  for img,mean,std in zip(image,all_means,all_stds):
    img=img.mul_(std).add_(mean)
  return image

def visualise(original_images,transformed_images):
  orig,transf=0,0
  figure= plt.figure(figsize=(8,8))
  for i in range(1,17):
    plt.subplot(4,4,i)
    plt.axis('off')
    plt.tight_layout()
    if orig<=transf:
      image=original_images[orig]
      plt.imshow(denormalise(image).permute(1,2,0))
      orig+=1
    else:
      image= transformed_images[transf]
      plt.imshow(denormalise(image).permute(1,2,0))
      transf+=1

def visualise_transformation():

  train_original = CustomImageDataset(dataframe=labels_df, img_dir='train/images/', transform=test_transforms)
  train_orig_loader = DataLoader(train_original, batch_size=64, shuffle=False)
  original_images, labels= next(iter(train_orig_loader))

  # Create datasets
  train_transformed = CustomImageDataset(dataframe=labels_df, img_dir='train/images/', transform=train_transforms)
  transformed_loader = DataLoader(train_transformed, batch_size=64, shuffle=False)
  transformed_images, labels= next (iter(transformed_loader))

  # plot the images
  return visualise(original_images,transformed_images)



