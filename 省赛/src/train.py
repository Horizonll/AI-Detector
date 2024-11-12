import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from model import *
from config import *
import argparse

root_dir = "/root/autodl-tmp/face/images"
dataset = AIDataset(root_dir=root_dir, transform=data_transforms["train"])
# val_dataset = AIDataset(root_dir=root_dir, transform=data_transforms["test"])
# dataset = torch.utils.data.ConcatDataset([dataset])
total_length = len(dataset)
train_length = int(0.5 * total_length)
val_length = total_length - train_length
train_dataset, val_dataset = random_split(dataset, [train_length, val_length])
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=16)


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Training Loss: {epoch_loss:.4f}")


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            predicted_labels = (outputs > 0.5).float()
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct_predictions / total_predictions
    print(f"Validation Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI Model")
    parser.add_argument(
        "--r", action="store_true", help="Resume training from checkpoint"
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AIModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    if args.r:
        model.load_state_dict(torch.load("model.pth"))
        print("Resumed training from checkpoint.")
    num_epochs = 20
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train(model, train_loader, criterion, optimizer, device)
        validate(model, val_loader, criterion, device)
        scheduler.step()
        torch.save(model.state_dict(), "model.pth")
