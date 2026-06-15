"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import json

import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loaders
import models
from fit import Trainer

def main():   
    with open("config.json", "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training executing on device: {device}")

    train_loader, val_loader, test_loader = get_loaders(
        data=config["DATA"], 
        data_path=config["DATA_PATH"], 
        batch_size=config["BATCH_SIZE"]
    )

    model_class = getattr(models, config["MODEL"])
    model = model_class(
        in_channels=config["CHANNELS"], 
        num_classes=config["NUM_CLASSES"], 
        drop_rate = config.get("DROP_RATE", 0.5),
        activation_str = config.get("ACTIVATION", "ReLU")
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

    trainer = Trainer(model, criterion, optimizer, device)
    trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"])
    
    
    test_loss, test_acc, prec, rec, f1 = trainer.evaluate(test_loader)
    print(f"\n  TEST → Loss: {test_loss:.4f} | "
            f"Acc: {test_acc:.2f}% | "
            f"P: {prec:.4f} | R: {rec:.4f} | F1: {f1:.4f}")


if __name__ == "__main__":
    main()