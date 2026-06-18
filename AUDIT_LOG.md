# AUDIT_LOG.md

## Post-Incident Pipeline Reconstruction Audit Log

---

| # | File | How the problem manifests | Mathematical / logical root cause | Structural correction implemented | Git commit hash |
|---|------|--------------------------|-----------------------------------|-----------------------------------|-----------------|
| 1 | `config.json` | Script crashes with `JSONDecodeError` on startup | `config.json` did not exist. `json.load()` on a missing file raises an immediate exception before anything runs | Created `config.json` with all required keys: | e5192a0 |
| 2 | `data.py` | Validation accuracy is inflated — metrics are unreliable | `train_data` was assigned the full array without slicing. Both train and val sets contained the same samples | Sliced `train_data` and `train_labels` to `[:val_start]` to make the two splits mutually exclusive | 1d1575e |
| 3 | `models.py` | `TypeError` when computing loss — `ResNet18` cannot train at all | Last line of `ResNet18.forward()` was `self.classifier(out)` with no `return`. Method returned `None` implicitly. Loss function received `None` and crashed immediately | Added `return` to the final line: `return self.classifier(out)` | 62094cd |
| 4 | `models.py` | `RuntimeError` when building model — `AlexNet` ignores `in_channels` and `num_classes` from config | `AlexNet.__init__` accepted only `**kwargs`. `in_channels` was hardcoded as `3` and `num_classes` as `11`. Config values were silently ignored for any other dataset | Added explicit `in_channels` and `num_classes` parameters to `AlexNet.__init__` matching the interface of `VGG16` and `ResNet18` | 34b2a2b  |
| 5 | `models.py` | `RuntimeError: mat1 and mat2 shapes cannot be multiplied` at Linear layer in `VGGBlock` | `padding=1` default applied to all convolutions including 1×1 tail conv. Applying `padding=1` to a 1×1 kernel inflates spatial dimensions causing shape mismatch at the classifier | Computed padding per convolution inside the loop: `padding = 0` for 1×1 convolutions and `padding = 1` for 3×3 convolutions | dc5a31d |
| 6 | `models.py` | `RuntimeError` — channel mismatch inside `VGGBlock` for blocks with more than one conv | `current_in_channels` was never updated inside the loop. Every convolution after the first received the original `in_channels` instead of `out_channels` | Added `current_in_channels = out_channels` at the end of each loop iteration | 96a9703 |
| 7 | `models.py` | `RuntimeError: mat1 and mat2 shapes cannot be multiplied` in `AlexNet` | `nn.Linear(2048, 1024)` was hardcoded. `AlexNet` last conv outputs `192` channels at `4×4` spatial size giving `192×4×4=3072`. `2048/192=10.67` is not a whole number — impossible at any input size | Changed `nn.Linear(2048, 1024)` to `nn.Linear(3072, 1024)` to match the actual flattened feature map size | 3627a91 |
| 8 | `data.py` | `ValueError` — labels shape `(N,1)` crashes `CrossEntropyLoss` and `torch.bincount` | Labels stored as shape `(N,1)` in `.pt` files. Both functions require flat `(N,)` int64 tensor. Passing `(N,1)` directly causes dimension mismatch | Added `.squeeze().long()` to all label tensors in `get_loaders` — train, val and test | 1bed2b2 |
| 9 | `fit.py` | `ValueError` — labels shape `(N,1)` crashes inside training and evaluation loops | Same label shape problem as bug 8 but occurring inside `Trainer` loops when labels come from `DataLoader` batches | Added `.squeeze().long()` to labels in both `train_one_epoch` and `evaluate` loops | ef7e0e9 |
| 10 | `fit.py` | Loss diverges or NaN from first epoch — all weight updates wrong | `optimizer.zero_grad()` missing — PyTorch accumulates gradients by default, each backward pass adds to previous gradients causing gradient explosion | Added `self.optimizer.zero_grad()` before forward pass inside training loop | 47dfae8 |
| 11 | `models.py` | ResNet18 trains without crashing but loss does not improve — model learns nothing | `activation_str` was a hardcoded module-level global placeholder — even if set to a valid activation like ReLU, hardcoding globally prevents flexible per-model configuration. With multiple models each needing independent activation control, a global creates confusion and unintended sharing | Removed global and read `activation_str` from `kwargs.get("activation_str", "ReLU")` inside `ResNet18.__init__` so each model instance can be configured independently at construction time | 2d1955a |
| 12 | `train.py` | Model trains but accuracy is flat — network capacity is destroyed | `drop_rate=0.99` and `activation_str=None` were hardcoded. `0.99` dropout zeros 99% of neurons every forward pass. `None` passed to `getattr(nn, None)` would crash if ever read | Replaced both with config reads: `drop_rate=config.get("DROP_RATE", 0.5)` and `activation_str=config.get("ACTIVATION", "ReLU")` | cb3f69f |
| 13 | `fit.py` | Variable `sum` silently overwrites Python built-in `sum()` in `train_one_epoch` | Counter variable was named `sum` shadowing the Python built-in for the entire scope. Inconsistent with `evaluate()` which already used `total` | Renamed variable from `sum` to `total` throughout `train_one_epoch` | 8a5aad4 |
| 14 | `models.py` | `AlexNet` training unstable — gradient magnitudes inconsistent across layers | First two conv layers had `BatchNorm2d`. Last three had none. Inconsistent normalisation causes uneven gradient flow across the feature extractor | Added `BatchNorm2d` after each of the three middle conv layers to make normalisation consistent throughout | d8c399e |
| 15 | `train.py` | No test metrics produced after training — pipeline cannot report required results | `get_loaders()` returns three loaders but the third was discarded with `_`. No test evaluation ever ran. Spec requires accuracy, precision, recall and F1 on test set | Changed to `train_loader, val_loader, test_loader = get_loaders(...)` and added test evaluation after training | 0465e11 |
| 16 | `fit.py` | `evaluate()` returns only loss and accuracy — spec requires four metrics | Original returned `(running_loss / total, (correct / total) * 100)`. Spec explicitly requires accuracy, precision, recall and macro F1. `train.py` unpacking 5 values would also crash | Added `all_preds` and `all_labels` collectors and computed precision, recall and F1 using sklearn after the loop | 1ca2953 |
| 17 | `train.py` | all dataset loop loads same dataset — wrong results for all runs | `get_loaders` called with hardcoded `config["DATA"]` instead of loop variable — every iteration loaded the same dataset ignoring the loop | Replaced `config["DATA"]` with `dataset_name` so each loop iteration loads the correct dataset | 9390ac0 |
| 18 | `models.py` | VGGBlock receives padding as external parameter — each conv layer may need different padding | Passing a single padding value externally forces all conv layers in the block to use the same padding, preventing per-layer flexibility. Padding should be computed internally from kernel size so each conv gets the correct value automatically | Removed padding parameter from VGGBlock interface and computed internally as `kernel_size // 2` per conv layer | 9bca7b5 |
| 19 | `fit.py` | Training crashes every epoch — val metrics never printed | `evaluate()` returns 5 values but `fit()` unpacked only 2. Python raises `ValueError: too many values to unpack` on the first validation call | Changed `val_loss, val_acc = self.evaluate(val_loader)` to `val_loss, val_acc, *_ = self.evaluate(val_loader)` | 359acb7 |





## Enhancement commits

| # | File | What was added | Reason | Git commit hash |
|---|------|---------------|--------|-----------------|
| 20 | `train.py` | Loop over all datasets and models from config | Original trained only one dataset/model per run. Full benchmark requires 15 runs across 5 datasets and 3 models | b4cbab4 |
| 21 | `train.py` | `compute_class_weights` — inverse frequency weighting | `lesions` class 5 is 67% of training data. Uniform loss causes model to predict majority class. Weighted loss forces equal attention to all classes | ca58386 |
| 22 | `train.py` | `save_result` — append results to CSV after every run | Results existed only in memory. Colab disconnect or session end lost everything. Persistent CSV saves each run immediately to Drive | 91a8342 |
| 23 | `train.py` | `already_done` — skip completed runs on restart | No way to resume after interruption. Checking for saved weight file allows training to continue from where it stopped across multiple sessions | 84f52f9 |



*Course: MAI/IDL SS26 — Final Assignment*
*Repository: https://github.com/hadera6/idl26_finalAssignment_Hadera_and_Kevin.git*
