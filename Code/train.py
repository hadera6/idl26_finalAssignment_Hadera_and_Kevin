"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import json
import os
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loaders
import models
from fit import Trainer

def compute_class_weights(train_labels, num_classes, device):
    """Inverse frequency weights to handle class imbalance."""
    train_labels = train_labels.squeeze().long()
    counts  = torch.bincount(train_labels, minlength=num_classes).float()
    counts  = counts.clamp(min=1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return weights.to(device)

def save_result(results_path, row):
    """Append one result row to benchmark CSV."""
    fieldnames = ['dataset', 'model', 'test_acc',
                  'precision', 'recall', 'f1']
    exists = os.path.exists(results_path)
    with open(results_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)
def already_done(weights_dir, dataset_name, model_name):
    """Return True if weights for this run already exist on disk."""
    return os.path.exists(
        os.path.join(weights_dir, f"{dataset_name}_{model_name}.pth")
    )


def main():   
    with open("config.json", "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training executing on device: {device}")

    weights_dir  = config["WEIGHTS_PATH"]
    results_path = config["RESULTS_PATH"]
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)

    dataset_configs = config["DATASETS"]
    model_names     = config["MODELS"]
    total_runs      = len(dataset_configs) * len(model_names)
    run_num         = 0

    for dataset_name, ds_cfg in dataset_configs.items():
        channels    = ds_cfg["CHANNELS"]
        num_classes = ds_cfg["NUM_CLASSES"]

        print(f"\n{'='*60}")
        print(f"  Loading: {dataset_name.upper()}")

        train_loader, val_loader, test_loader = get_loaders(
            data=config["DATA"], 
            data_path=config["DATA_PATH"], 
            batch_size=config["BATCH_SIZE"]
        )

        train_labels_all = torch.cat(
            [y for _, y in train_loader]).squeeze().long()
        class_weights = compute_class_weights(
            train_labels_all, num_classes, device)
        print(f"  Class weights: {class_weights.cpu().numpy().round(3)}")

        for model_name in model_names:
            run_num += 1
            print(f"\n  [{run_num}/{total_runs}] {model_name} on {dataset_name}")
            print(f"  {'-'*50}")

            if already_done(weights_dir, dataset_name, model_name):
                print("  Already done — skipping.")
                continue

            model_class = getattr(models, model_name)
            model = model_class(
                in_channels    = channels,
                num_classes    = num_classes,
                    
                drop_rate      = config.get("DROP_RATE", 0.5),
                activation_str = config.get("ACTIVATION", "ReLU")
            ).to(device)

            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

            trainer = Trainer(model, criterion, optimizer, device)
            trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"])
            
            
            test_loss, test_acc, prec, rec, f1 = trainer.evaluate(test_loader)
            print(f"\n  TEST → Loss: {test_loss:.4f} | "
                    f"Acc: {test_acc:.2f}% | "
                    f"P: {prec:.4f} | R: {rec:.4f} | F1: {f1:.4f}")

            weight_path = os.path.join(
                weights_dir, f"{dataset_name}_{model_name}.pth")
            torch.save(model.state_dict(), weight_path)
            print(f"  Weights saved → {weight_path}")

            # FIX 19 — save result immediately after each run
            save_result(results_path, {
                'dataset':   dataset_name,
                'model':     model_name,
                'test_acc':  round(test_acc, 2),
                'precision': round(prec, 4),
                'recall':    round(rec, 4),
                'f1':        round(f1, 4)
            })

            del model
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("All runs complete!")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()