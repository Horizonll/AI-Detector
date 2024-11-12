import os

# os.system("pip install -r requirements.txt")
import torch
from torch.utils.data import DataLoader
import pandas as pd
from config import *


def predict(model, dataloader, device):
    model.eval()
    predictions = []
    filenames = []
    with torch.no_grad():
        for inputs, names in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.float().squeeze().cpu().numpy()
            predictions.extend(preds)
            filenames.extend(names)
    return filenames, predictions


def save_results(filenames, predictions, output_path):
    results = pd.DataFrame({"id": filenames, "label": predictions})
    results["label"] = results["label"].apply(lambda x: 1 if x >= 0.5 else 0)
    results.sort_values(by="id", inplace=True)
    results.to_csv(output_path, index=False, header=False)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir = "/testdata"
    test_dataset = TestDataset(test_dir, transform=data_transforms["test"])
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=16
    )
    model_path = "model.pth"
    model = torch.load(model_path).to(device)
    filenames, predictions = predict(model, test_loader, device)
    output_path = "../cla_pre.csv"
    save_results(filenames, predictions, output_path)
