import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler 
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
from torchvision.models import resnet18, ResNet18_Weights
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
import wandb
from torchmetrics import AUROC, F1Score
torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

wandb.login()

train_dir = 'imagenet-combine-train/train/'
valid_dir = 'imagenet-combine-train/val/'
save_path = 'model_ckpt_4/model'

class DILLEMADataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
   
    @property
    def classes(self):
        return self.data.classes
    
    @property
    def imgs(self):
        return self.data.imgs
    
    @property
    def class_to_idx(self):
        return self.data.class_to_idx


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_train = DILLEMADataset(
    data_dir=train_dir,
    transform=preprocess
)
data_valid = DILLEMADataset(
    data_dir=valid_dir,
    transform=preprocess
)

train_dataloader = DataLoader(data_train, batch_size=100, num_workers=8, shuffle=True)
valid_dataloader = DataLoader(data_valid, batch_size=100, num_workers=8, shuffle=False)


weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

for param in model.parameters():
    param.requires_grad = True


def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)
    

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
criterion = nn.CrossEntropyLoss()
f1 = F1Score(task="multiclass", num_classes=len(data_train.classes))

#### Train model
train_loss = []
train_accuracy = []
valid_loss = []
valid_accuracy = []

# wandb.init(id='olive-sponge-30', resume="must")

wandb.init(
    project="DILLEMA-3", 
    config={"architecture": "ResNet18",
            "dataset": "Imagenet1K",
            "epochs": 90,
      })


num_epochs = 90   #(set no of epochs)
start_time = time.time() #(for showing time)
model.to(device)
# Start loop
for epoch in range(num_epochs): #(loop for every epoch)
    print("Epoch {} running".format(epoch)) #(printing message)
    """ Training Phase """
    model.train()    #(training model)
    running_loss = 0.   #(set loss 0)
    running_corrects = 0 
    predicts = []
    targets = []
    predicts_val = []
    targets_val = []
    # load a batch data of images
    for inputs, labels in tqdm(train_dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        # forward inputs and get output
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        # get loss value and update the network weights
        loss.backward()
        optimizer.step()
        predicts.append(preds.cpu())
        targets.append(labels.cpu())
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data).item()
    train_epoch_loss = running_loss / len(data_train)
    train_epoch_acc = running_corrects / len(data_train) * 100.
    preds_train_tensor = torch.cat(predicts)
    targets_train_tensor = torch.cat(targets)
    # Append result
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_acc)
    f1_score_train = f1(preds_train_tensor, targets_train_tensor)
    # Print progress
    print('[Train #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, train_epoch_loss, train_epoch_acc, time.time() - start_time))
    """ Validation Phase """
    model.eval()
    with torch.no_grad():
        running_loss = 0.
        running_corrects = 0
        for inputs, labels in tqdm(valid_dataloader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item()
            predicts_val.append(preds.cpu())
            targets_val.append(labels.cpu())
        valid_epoch_loss = running_loss / len(data_valid)
        valid_epoch_acc = running_corrects / len(data_valid) * 100.
        preds_valid_tensor = torch.cat(predicts_val)
        targets_valid_tensor = torch.cat(targets_val)
        # Append result
        valid_loss.append(valid_epoch_loss)
        valid_accuracy.append(valid_epoch_acc)
        f1_score_valid = f1(preds_valid_tensor, targets_valid_tensor)
        # Print progress
        print('[Valid #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, valid_epoch_loss, valid_epoch_acc, time.time() - start_time))
    wandb.log({"Train loss": train_epoch_loss, "Valid loss": valid_epoch_loss,
              "Train acc": train_epoch_acc, "Valid acc": valid_epoch_acc, 
               "f1_score_train": f1_score_train, "f1_score_valid": f1_score_valid,
               "epoch": epoch})
    # wandb.watch(model, criterion, log="all", log_freq=10)
    save_model = f"{save_path}_{epoch}.pt"
    save_checkpoint(model, optimizer, save_model, epoch)
    scheduler.step()