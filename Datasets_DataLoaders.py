import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ANNOTATIONS_FILE = "./files/fashion-mnist_test.csv"


# custom dataset settings
# csv download link：https://www.kaggle.com/zalando-research/fashionmnist
class CustomImageDataset(Dataset):
    # the input of __init__ are the same input as we init the class
    def __init__(self, annotations_file, transform=None):
        self.data_path = pd.read_csv(annotations_file)
        self.transform = transform
        self.images = self.data_path.iloc[:,1:].values.astype(np.uint8)
        self.labels = self.data_path.iloc[:, 0].values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # get the image and the lable
        # images and lables are in seprate lists, use "idx" as index to make sure they are match
        image = self.images[idx].reshape(28,28,1)
        label = int(self.labels[idx])

        # set the transform settings
        # make sure image and label are "tensor" type both
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image/255., dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)

        # return "image, label" as the return of the dataset class
        return image, label

# data transforms
image_size = 28
data_transform = transforms.Compose([
    transforms.ToPILImage(),  
    transforms.Resize(image_size),
    transforms.ToTensor()
])

# load the data
test_data = CustomImageDataset(ANNOTATIONS_FILE, data_transform)
test_dataloader = DataLoader(test_data, batch_size=64)


# Display image and label.

# extract a batch of data using a Iterater, and value them to test_features, test_labels respectly
test_features, test_labels = next(iter(test_dataloader))
print(f"Feature batch shape: {test_features.size()}")
print(f"Labels batch shape: {test_labels.size()}")

# get the first pair of data
# the squeeze() function is to remove any singleton dimensions
img = test_features[0].squeeze()
label = test_labels[0]

# show the image
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

