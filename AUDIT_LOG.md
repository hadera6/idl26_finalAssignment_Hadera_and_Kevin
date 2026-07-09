# AUDIT LOG
## Pipeline Fix and Enhancement Registry

### Bug Fixes

> **Note:** The codebase contained 15–17 distinct issues resolved across 20 commits. The higher commit count reflects fixes that were incomplete on the first attempt and required follow-up commits to fully resolve.

> Entries 1–20 are bug fixes. Entries 21 onwards are enhancements new features added to the pipeline such as benchmarking, profiling, lighter models, and transfer learning.

| # | File | How the Problem Manifests | Root Cause | Fix applied | Commit |
|---|------|--------------------------|------------|-----|--------|
| 1 | `config.json` | Script crashes with `JSONDecodeError` on startup before anything runs | `config.json` did not exist — `json.load()` on a missing file raises an immediate exception | Created `config.json` with all required keys | e5192a0 |
| 2 | `data.py` | Validation accuracy inflated — metrics unreliable across all runs | `train_data` assigned the full array without slicing — both train and val splits contained identical samples | Sliced `train_data` and `train_labels` to `[:val_start]` | 1d1575e |
| 3 | `models.py` | `TypeError` when computing loss — `ResNet18` cannot train at all | Last line of `ResNet18.forward()` was `self.classifier(out)` with no `return` — method returned `None` implicitly; loss function received `None` and crashed | Added `return` to the final line | 62094cd |
| 4 | `models.py` | Model builds without error but `in_channels` and `num_classes` from config are silently ignored | `AlexNet.__init__` accepted only `**kwargs` — `in_channels` hardcoded as `3`, `num_classes` as `11`; any other dataset silently used wrong values | Added explicit `in_channels` and `num_classes` parameters matching `VGG16` and `ResNet18` interface | 34b2a2b |
| 5 | `models.py` | `RuntimeError: mat1 and mat2 shapes cannot be multiplied` at the Linear layer | `padding=1` default applied to all convolutions including the 1×1 tail conv — applying `padding=1` to a 1×1 kernel inflates spatial dimensions causing shape mismatch at the classifier | Computed padding per conv inside the loop: `0` for 1×1 kernels, `1` for 3×3 | dc5a31d |
| 6 | `models.py` | `RuntimeError` — channel mismatch on second and subsequent convolutions inside any `VGGBlock` with more than one conv | `current_in_channels` never updated inside the loop — every convolution after the first received the original `in_channels` instead of `out_channels` | Added `current_in_channels = out_channels` at end of each loop iteration | 96a9703 |
| 7 | `models.py` | `RuntimeError: mat1 and mat2 shapes cannot be multiplied` at `AlexNet` classifier | `nn.Linear(2048, 1024)` was hardcoded — the last conv outputs 192 channels at 4×4 giving `192×4×4=3072`; 2048 is impossible at any valid input size | Changed to `nn.Linear(3072, 1024)` | 3627a91 |
| 8 | `data.py` | `ValueError` in `CrossEntropyLoss` and `torch.bincount` — training cannot start | Labels stored as shape `(N,1)` in `.pt` files; both functions require flat `(N,)` int64 tensor — passing `(N,1)` causes a dimension mismatch | Added `.squeeze().long()` to all label tensors in `get_loaders` | 1bed2b2 |
| 9 | `fit.py` | Same `ValueError` as bug 8 reappears inside training and evaluation loops even after bug 8 fix | Same root cause — label shape not corrected inside `Trainer` when labels arrive from `DataLoader` batches | Added `.squeeze().long()` to labels in both `train_one_epoch` and `evaluate` | ef7e0e9 |
| 10 | `fit.py` | Loss diverges or produces NaN from the first epoch — all weight updates are corrupted | `optimizer.zero_grad()` missing — PyTorch accumulates gradients by default; each backward pass adds to previous batch gradients causing explosion | Added `self.optimizer.zero_grad()` before forward pass in training loop | 47dfae8 |
| 11 | `models.py` | `ResNet18` trains without crashing but loss does not improve — model learns nothing regardless of dataset | `activation_str` was a hardcoded module-level global placeholder — even setting it externally had no effect since each instance ignored it | Removed global; read `activation_str` from `kwargs.get("activation_str", "ReLU")` inside `__init__` so each instance is independently configurable | 2d1955a |
| 12 | `train.py` | Model trains but accuracy stays flat — no architecture or dataset produces improvement | `drop_rate=0.99` hardcoded — 99% of neurons zeroed every forward pass destroying capacity; `activation_str=None` would crash if evaluated | Replaced both with config reads: `ds_cfg.get("DROP_RATE", 0.5)` and `ds_cfg.get("ACTIVATION", "ReLU")` | cb3f69f |
| 13 | `fit.py` | Loss accumulation silently produces wrong values throughout training | Variable named `sum` shadowed Python built-in `sum()` for the entire `train_one_epoch` scope — inconsistent with `evaluate()` which already used `total` | Renamed from `sum` to `total` throughout `train_one_epoch` | 8a5aad4 |
| 14 | `models.py` | `AlexNet` training unstable — loss spikes and slow convergence even on simple datasets | First two conv layers had `BatchNorm2d`; last three had none — inconsistent normalisation causes uneven gradient magnitudes across the feature extractor | Added `BatchNorm2d` after each of the three middle conv layers | d8c399e |
| 15 | `train.py` | Training completes without error but no test metrics are ever produced or saved | `get_loaders()` returns three loaders but the third was discarded with `_` — test evaluation never ran | Changed to `train_loader, val_loader, test_loader = get_loaders(...)` and added test evaluation call after training | 0465e11 |
| 16 | `fit.py` | Only loss and accuracy available — precision, recall and F1 missing from all outputs | `evaluate()` returned only `(loss, accuracy)` — spec requires four metrics; `train.py` unpacking 5 values would also crash | Added `all_preds` and `all_labels` collectors; computed precision, recall and F1 via sklearn after the loop | 1ca2953 |
| 17 | `train.py` | All dataset loop iterations load and train on the same single dataset — wrong results for every run | `get_loaders` called with hardcoded `config["DATA"]` instead of the loop variable `dataset_name` | Replaced `config["DATA"]` with `dataset_name` | 9390ac0 |
| 18 | `models.py` | `VGGBlock` applies the same padding to all conv layers regardless of kernel size — incorrect for blocks with mixed kernel sizes | Padding was passed as a single external parameter forcing all layers to use the same value; 1×1 convolutions require `padding=0` not `padding=1` | Removed padding parameter from `VGGBlock` interface; computed internally as `kernel_size // 2` per conv | 9bca7b5 |
| 19 | `fit.py` | `ValueError: too many values to unpack` crashes training on the first validation call | `fit()` unpacked only 2 values from `evaluate()` which now returns 5 after bug 16 was fixed | Changed to `val_loss, val_acc, *_ = self.evaluate(val_loader)` | 359acb7 |
| 20 | `models.py` | Severe train-test gap — model appears to memorise training image positions rather than visual content | No `AdaptiveAvgPool2d` — classifier received full spatial feature map `192×4×4` in AlexNet and `512×2×2` in VGG16; model learned where features appear rather than what they are | Added `self.avgpool = nn.AdaptiveAvgPool2d((1,1))` to `AlexNet` and `VGG16`; corrected Linear input size to match pooled output | cd04476 |

---

### Enhancements

| # | File | What was added | Reason | Commit |
|---|------|---------------|--------|--------|
| 21 | `train.py` | Loop over all datasets and models from config | Original trained only one dataset/model per run | b4cbab4 |
| 22 | `train.py` | `compute_class_weights` — inverse frequency weighting | Imbalanced datasets predicted majority class only | ca58386 |
| 23 | `train.py` | `save_result` — append results to CSV after every run | Results lost on Colab disconnect | 91a8342 |
| 24 | `train.py` | `already_done` — skip completed runs on restart | No way to resume after interruption | 84f52f9 |
| 25 | `train.py` | Weighted loss removed | Showed inconsistent benefit — removed to isolate effect | ae6541b |
| 26 | `data.py` | Normalisation added | Raw pixel values caused unstable training and loss spikes | 32e8379 |
| 27 | `train.py` | Per-dataset hyperparameter control | Global config only — per-dataset tuning impossible | 13004aa |
| 28 | `train.py` `fit.py` | Learning rate scheduling | Fixed LR caused late-epoch bouncing | 23cd076 |
| 29 | `train.py` `fit.py` | Early stopping with best weight restoration | Training ran full epochs regardless of convergence | adae156 |
| 30 | `train.py` | `compute_class_weights` re-added | Removed in #25 — reintroduced after targeted testing | eb30e5f |
| 31 | `models.py` | `AlexNetLite` added | Full AlexNet at 2.7M params excessive for deployment | ce65003 |
| 32 | `train.py` | Resource profiling — runtime, latency, peak memory | No visibility into computational cost per model | c73cdb9 |
| 33 | `train.py` `data.py` | Seed control for reproducibility | Results varied between runs with identical config | 71467b0 |
| 34 | `train.py` | Iterator exhaustion fix | Shared train loader consumed before model loop — subsequent models received empty data | c1c9223 |
| 35 | `train.py` | CUDA memory reset guarded by device check | Crashed on CPU runtime | 0ac2a06 |
| 36 | `train.py` | CUDA peak memory query guarded by device check | Crashed on CPU runtime | 1772c8d |
| 37 | `models.py` | `VGG16Lite` added | Full VGG16 at 11M params excessive for deployment | 8163f1a |
| 38 | `models.py` | `ResNet18Lite` added | Full ResNet18 at 11M params excessive for deployment | df95fc2 |
| 39 | `config.json` | Per-dataset MODELS arrays, hyperparameter overrides, transfer learning keys | Global config structure could not support per-dataset tuning or transfer learning | 76af0cb |
| 40 | `train.py` | Per-dataset MODELS reading and total_runs recomputation | Global MODELS list applied same models to all datasets | 766009f |
| 41 | `train.py` | `load_pretrained_weights()` and `freeze_stages()` added | No mechanism to load weights from another dataset's checkpoint | bf9d0e0 |
| 42 | `train.py` | Transfer learning integrated into main training loop | Functions defined but never called | 5818160 |
| 43 | `train.py` | Training state tracking in weights and benchmark CSV | Scratch and transfer runs were indistinguishable | 358e928 |
| 44 | `data.py` | `AugmentedDataset` class and `augment` flag in `get_loaders` | No augmentation support — scarce datasets overfitted quickly | b5a7fb3 |
| 45 | `models.py` | `AlexNetMicro` added — 91,147 params (30× smaller) | AlexNetLite at ~400K exceeded requirements for simpler datasets | 9ab99d8 |
| 46 | `models.py` | `VGG16Micro` added — 296,571 params (37× smaller) | VGG16Lite at ~324K exceeded requirements for simpler datasets | 5c65c8b |
| 47 | `models.py` | `ResNet18Micro` added — 224,228 params (50× smaller) | ResNet18Lite at ~171K exceeded requirements for simpler datasets | 7bdbace |
| 48 | `train.py` | Per-dataset batch size support | Batch size hardcoded globally — could not vary by dataset | f8080f7 |
| 49 | `train.py` | `freeze_stages()` simplified — universal classifier-based freeze | Architecture-specific freezing provided no measurable accuracy benefit | f65e493 |
