import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from model import AIModel, BaggingModel
from config import *
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

root_dir = "/root/autodl-tmp/c"
train_dataset = AIDataset(root_dir=root_dir, transform=data_transforms["train"])
val_dataset = train_dataset
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=16)
current_time = datetime.now().strftime("%d-%H:%M:%S")
log_dir = f"/root/tf-logs/{current_time}"
writer = SummaryWriter(log_dir)


def train(model, dataloader, criterion, optimizer, device, epoch, model_index):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(dataloader, desc=f"Training Model {model_index}"):
        inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    writer.add_scalar(f"Loss/Train_Model_{model_index}", epoch_loss, epoch + 1)
    print(f"Model {model_index} Training Loss: {epoch_loss:.5f}")


def validate(models, dataloader, criterion, device, epoch):
    bagging_model = BaggingModel(models).to(device)
    bagging_model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
            outputs = bagging_model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            predicted_labels = (outputs > 0.5).float()
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct_predictions / total_predictions
    writer.add_scalar("Loss/Validation", epoch_loss, epoch + 1)
    writer.add_scalar("Accuracy/Validation", accuracy, epoch + 1)
    print(f"Validation Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}\n")
    return bagging_model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [
        AIModel(efficientnet_type="efficientnet-b0").to(device),
        AIModel(efficientnet_type="efficientnet-b0").to(device),
        AIModel(efficientnet_type="efficientnet-b1").to(device),
        AIModel(efficientnet_type="efficientnet-b1").to(device),
        AIModel(efficientnet_type="efficientnet-b1").to(device),
    ]
    train_loaders = []
    optimizers = []
    schedulers = []
    criterions = []
    for i in range(len(models)):
        sampler = RandomSampler(train_dataset, replacement=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            sampler=sampler,
            num_workers=16,
        )
        train_loaders.append(train_loader)
        optimizer = optim.AdamW(models[i].parameters())
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        criterion = nn.BCELoss()
        criterions.append(criterion)
    num_epochs = 100
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for i, model in enumerate(models):
            train(
                model,
                train_loaders[i],
                criterions[i],
                optimizers[i],
                device,
                epoch,
                i + 1,
            )
            schedulers[i].step(epoch + epoch / len(train_loaders[i]))
        if (epoch + 1) % 10 == 0:
            bagging_model = validate(models, val_loader, criterions[0], device, epoch)
    bagging_model = BaggingModel(models).to(device)
    torch.save(bagging_model, "model.pth")
    writer.close()
