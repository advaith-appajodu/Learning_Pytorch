import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# !pip install torchvision
import torchvision

import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms



transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),   
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

train_data = datasets.CIFAR10(root="/data", train=True, download=True, transform=transform_train)
test_data = datasets.CIFAR10(root="/data", train=False, download=True, transform=transform_test)


batch_size = 256
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class CNN_GAP(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, dropout_p=0.3):
        super(CNN_GAP, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x8x8
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128x4x4
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Output: 256x1x1
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

def imshow(img):
   npimg = img.numpy()
   plt.imshow(np.transpose(npimg, (1, 2, 0)))
   plt.show()

# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
labels
# show images
imshow(torchvision.utils.make_grid(images))


model = CNN_GAP().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    correct, total_loss, total = 0, 0, 0
    
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
        
        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {100*correct/total:.2f}%")
    
    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    print(f"Train Epoch: Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%\n")
    return avg_loss, accuracy


def test(model, loader, loss_fn):
    model.eval()
    correct, total_loss, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    print(f"Test: Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%\n")
    return avg_loss, accuracy



num_epochs = 225
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_one_epoch(model, train_loader, loss_fn, optimizer)
    test(model, test_loader, loss_fn)

torch.save(model.state_dict(), "CIFAR-10_82_CNN.pth")