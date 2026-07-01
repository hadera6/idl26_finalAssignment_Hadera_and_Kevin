"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import json
import os
import csv
import time

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

def save_profiling(profiling_path, row):                                       
    """Append one profiling row to profiling CSV."""
    fieldnames = ['dataset', 'model', 'train_time_s',
                  'latency_ms', 'peak_memory_mb']
    exists = os.path.exists(profiling_path)
    with open(profiling_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():   
    with open("config.json", "r") as f:
        config = json.load(f)

    seed = config.get("SEED", 0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training executing on device: {device}")


    weights_dir    = config["WEIGHTS_PATH"]
    results_path   = config["RESULTS_PATH"]
    profiling_path = config["PROFILING_PATH"]

    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    os.makedirs(os.path.dirname(profiling_path), exist_ok=True)

    dataset_configs = config["DATASETS"]
    total_runs = sum(len(ds_cfg["MODELS"]) for ds_cfg in dataset_configs.values())
    run_num         = 0

    for dataset_name, ds_cfg in dataset_configs.items():
        channels    = ds_cfg["CHANNELS"]
        num_classes = ds_cfg["NUM_CLASSES"]

        print(f"\n{'='*60}")
        print(f"  Loading: {dataset_name.upper()}")

        # Call 1 — temp load for label extraction only
        tmp_loader, _, _ = get_loaders(
            data=dataset_name, 
            data_path=config["DATA_PATH"],
            batch_size=config["BATCH_SIZE"],
            seed=seed
        )
        train_labels_all = torch.cat(
            [y for _, y in tmp_loader]).squeeze().long()
        class_weights = compute_class_weights(train_labels_all, num_classes, device)
        
        print(f"  Class weights: {class_weights.cpu().numpy().round(3)}")
        
        model_names = ds_cfg["MODELS"]

        for model_name in model_names:

            train_loader, val_loader, test_loader = get_loaders(
              data=dataset_name, 
              data_path=config["DATA_PATH"],
              batch_size=config["BATCH_SIZE"],
              seed=seed
            )
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
                    
                drop_rate      = ds_cfg.get("DROP_RATE",     config.get("DROP_RATE", 0.5)),
                activation_str = ds_cfg.get("ACTIVATION",    config.get("ACTIVATION", "ReLU"))
            
            ).to(device)

            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.Adam(
                model.parameters(), 
                lr=ds_cfg.get("LEARNING_RATE", config["LEARNING_RATE"])
            )

            epochs        = ds_cfg.get("EPOCHS", config["EPOCHS"])

            use_scheduler = ds_cfg.get("USE_SCHEDULER", True)                 
            scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs) if use_scheduler else None 

            patience   = ds_cfg.get("PATIENCE",  config.get("PATIENCE", 10))
            no_restore = ds_cfg.get("NO_RESTORE", False)                        

            trainer = Trainer(model, criterion, optimizer, device)

            if device.type == 'cuda':                          
              torch.cuda.reset_peak_memory_stats()                               
            train_start = time.time()

            trainer.fit(
                train_loader, val_loader, 
                epochs=epochs,
                scheduler  = scheduler,
                patience   = patience,                                         
                no_restore = no_restore
            )


            train_time = time.time() - train_start                            

            # inference latency — average per sample over one test batch
            model.eval()
            with torch.no_grad():
                batch_images, _ = next(iter(test_loader))
                batch_images = batch_images.to(device)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.time()
                _ = model(batch_images)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                latency_ms = (time.time() - t0) / len(batch_images) * 1000    

            if device.type == 'cuda':
              peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
            else:
              peak_memory_mb = 0.0
            
            
            test_loss, test_acc, prec, rec, f1 = trainer.evaluate(test_loader)
            print(f"\n  TEST → Loss: {test_loss:.4f} | "
                    f"Acc: {test_acc:.2f}% | "
                    f"P: {prec:.4f} | R: {rec:.4f} | F1: {f1:.4f}")

            weight_path = os.path.join(
                weights_dir, f"{dataset_name}_{model_name}.pth")
            torch.save(model.state_dict(), weight_path)
            print(f"  Weights saved → {weight_path}")
            save_result(results_path, {
                'dataset':   dataset_name,
                'model':     model_name,
                'test_acc':  round(test_acc, 2),
                'precision': round(prec, 4),
                'recall':    round(rec, 4),
                'f1':        round(f1, 4)
            })

            save_profiling(profiling_path, {                                   
                'dataset':        dataset_name,
                'model':          model_name,
                'train_time_s':   round(train_time, 2),
                'latency_ms':     round(latency_ms, 4),
                'peak_memory_mb': round(peak_memory_mb, 2)
            })

            del model
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("All runs complete!")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()