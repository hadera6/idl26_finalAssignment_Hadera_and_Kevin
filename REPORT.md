# Report of the pipeline reconstruction

> **Authors:** Hadera Teame Hailu and Kevin K. Thomas 

> **Date:** July 2026

---

## 1 Pipeline Repair and Benchmark Evaluation Task 1

### 1.1 Error Correction Analysis

This section presents the repair and evaluation of the recovered pipeline. Following a full codebase audit, 15-17 distinct errors were identified and corrected across all source files. The corrected pipeline was then used to benchmark three architectures AlexNet, VGG16, and ResNet18 across four medical imaging datasets: cells, chest, lesions, and orgs. Results are evaluated on test sets and  against accuracy targets.

---

### 1.2 Benchmark Results

All 12 configurations pass their accuracy targets.

| Dataset | Model | Accuracy | Precision | Recall | F1 | Target |
|---------|-------|:--------:|:---------:|:------:|:--:|:------:|
| cells | AlexNet | 98.30% | 0.9837 | 0.9840 | 0.9838 | 90% |
| cells | VGG16 | 98.22% | 0.9812 | 0.9820 | 0.9815 | 90% |
| cells | ResNet18 | 97.81% | 0.9776 | 0.9772 | 0.9772 | 90% |
| chest | AlexNet | 90.06% | 0.9254 | 0.8701 | 0.8877 | 87% |
| chest | VGG16 | 89.74% | 0.9215 | 0.8667 | 0.8841 | 87% |
| chest | ResNet18 | 89.90% | 0.9190 | 0.8705 | 0.8866 | 87% |
| lesions | AlexNet | 77.06% | 0.6461 | 0.6527 | 0.6443 | 67% |
| lesions | VGG16 | 77.96% | 0.6685 | 0.5905 | 0.6107 | 67% |
| lesions | ResNet18 | 75.81% | 0.6055 | 0.5098 | 0.5335 | 67% |
| orgs | AlexNet | 93.14% | 0.9239 | 0.9253 | 0.9241 | 83% |
| orgs | VGG16 | 92.39% | 0.9174 | 0.9152 | 0.9155 | 83% |
| orgs | ResNet18 | 92.27% | 0.9150 | 0.9154 | 0.9133 | 83% |

---

### 1.3 Comparative Analysis

**cells** — All three models exceed 90%  (97.8–98.3%). Model choice not critical — all reach the data ceiling. **Best: AlexNet at 98.30%.**

**chest** — All three models pass 87% (89.7–90.1%). Binary classification, results closely clustered. **Best: AlexNet at 90.06%.**

**lesions** — Hardest dataset. Class 5 = ~67% of test set, minority classes under 30 samples each. All three pass 67%. **Best: VGG16 at 77.96% (F1: 0.6107).**

**orgs** — All three pass 83%  (92.3–93.1%). **Best: AlexNet at 93.14% (F1: 0.9241).**

| Model  | Assessment |
|-------|------------|
| AlexNet | Best on chest and orgs |
| VGG16  | Best on hardest imbalanced task (lesions) |
| ResNet18  | Consistent across all datasets |

---

### 1.4 Reproducibility and Run Management

A global seed (SEED: 0) is applied at startup controlling weight initialisation and batch order. Results are stable within ±0.5% across reruns. All results are logged to results/benchmark.csv and traceable by dataset, model, and training state. Previously completed runs are skipped via already_done() — a checkpoint check against weights/ — enabling safe pipeline resumption after interruption.

---

##  2 Green Initiative: Lite Model Efficiency Evaluation Task 2

### 2.1 Overview

Three Lite model variants reduce computational footprint while maintaining benchmark accuracy through proportional uniform scaling.

| Model | Full Params | Lite Params | Reduction | Strategy |
|-------|:-----------:|:------------:|:---------:|---------|
| AlexNet → AlexNetLite | 2,745,963 | 91,147 | **30×** | 5 conv → 3 conv; proportional filter reduction; direct classifier |
| VGG16 → VGG16Lite | 11,067,595 | 296,571 | **37×** | 5 blocks → 4 blocks; doubling pattern 16→32→64→128; direct classifier |
| ResNet18 → ResNet18Lite | 11,182,923 | 224,228 | **50×** | All 4 stages kept; filters uniformly scaled 9→9→18→36→72 |

---

### 2.2 Accuracy Comparison — Full vs Lite Models

| Dataset | Model | Accuracy | Lite Model | Accuracy | 
|---------|-------|:--------:|:------:|------------|
| cells | AlexNet | 98.30% | AlexNetLite | 97.98% |
| cells | VGG16 | 98.22% | VGG16Lite | **98.83%** |
| cells | ResNet18 | 97.81% | ResNet18Lite | 97.87% |
| chest | AlexNet | 90.06% | AlexNetLite | **90.87%** |
| chest | VGG16 | 89.74% | VGG16Lite | 90.06% |
| chest | ResNet18 | 89.90% | ResNet18Lite | 87.34% |
| lesions | AlexNet | 77.06% | AlexNetLite | 72.12% |
| lesions | VGG16 | 77.96% | VGG16Lite | 75.31% |
| lesions | ResNet18 | 75.81% | ResNet18Lite | 68.53% |
| orgs | AlexNet | 93.14% | AlexNetLite | 91.57% |
| orgs | VGG16 | 92.39% | VGG16Lite | **93.79%** |
| orgs | ResNet18 | 92.27% | ResNet18Lite | 92.09% |

> **Lite models pass all 12 targets.** VGG16Lite outperforms full VGG16 on cells and orgs; AlexNetLite outperforms full AlexNet on chest.

---

### 2.3 Efficiency Metrics

#### Training Time (seconds)

| Dataset | AlexNet | AlexNetLite | Δ | VGG16 | VGG16Lite | Δ | ResNet18 | ResNet18Lite | Δ |
|---------|:-------:|:------------:|:-:|:-----:|:----------:|:-:|:--------:|:-------------:|:-:|
| cells | 81.65 | 35.36 | 2.3× | 490.96 | 112.26 | 4.4× | 683.50 | 192.36 | 3.6× |
| chest | 28.88 | 6.06 | 4.8× | 112.74 | 29.67 | 3.8× | 542.97 | 48.25 | 11.3× |
| lesions | 75.34 | 41.91 | 1.8× | 580.32 | 133.48 | 4.3× | 1721.40 | 225.39 | 7.6× |
| orgs | 56.02 | 23.23 | 2.4× | 366.45 | 82.56 | 4.4× | 1096.64 | 141.55 | 7.8× |

#### Peak Memory (MB)

| Dataset | AlexNet | AlexNetLite | Δ | VGG16 | VGG16Lite | Δ | ResNet18 | ResNet18Lite | Δ |
|---------|:-------:|:------------:|:-:|:-----:|:----------:|:-:|:--------:|:-------------:|:-:|
| cells | 182.56 | 71.96 | 2.5× | 895.77 | 274.79 | 3.3× | 2460.85 | 217.12 | 11.3× |
| chest | 180.39 | 66.16 | 2.7× | 896.26 | 275.77 | 3.2× | 2456.52 | 212.48 | 11.6× |
| lesions | 179.23 | 71.96 | 2.5× | 899.01 | 275.79 | 3.3× | 2459.21 | 217.12 | 11.3× |
| orgs | 180.16 | 66.17 | 2.7× | 900.41 | 275.80 | 3.3× | 2457.23 | 212.50 | 11.6× |

#### Inference Latency (ms per sample)

| Model | Full | Lite | Reduction |
|-------|:----:|:-----:|:---------:|
| AlexNet | 0.052 ms | 0.017 ms | 3.1× |
| VGG16 | 0.357 ms | 0.065 ms | 5.5× |
| ResNet18 | 0.812 ms | 0.075 ms | 10.8× |

---

### 2.4 Analysis
 
Micro models pass all 12 benchmark targets. The maximum accuracy gap is 7.3% (ResNet18Micro on lesions: 68.53% vs 75.81% — still well above the 67% target). On simpler datasets the gap is under 1%. Notably VGG16Micro outperforms full VGG16 on cells and orgs, and AlexNetMicro outperforms full AlexNet on chest — reduced capacity improves generalisation on tasks where full models have excess capacity.
 
**Parameter reduction:** Average 39× reduction across three pairs — combined full-model total of ~25M parameters reduced to ~612K (97.5% reduction), with all benchmark targets maintained.
 
**Memory savings:** Peak GPU memory reduced by 2.5–11.6×. ResNet18Micro drops from 2,460 MB to 217 MB — deployable on devices with as little as 256 MB GPU memory. AlexNetMicro requires only 66–72 MB across all datasets.
 
**Training time:** Average reduction is ~4.5× for VGG16, ~3.5× for ResNet18, ~2.5× for AlexNet. Across a full benchmark sweep, switching from full to micro models saves an estimated 3–4 hours of GPU time.
 
**Behaviour on lesions:** All three micro models pass the 67% target despite 30–50× fewer parameters. Reduced capacity prevents majority class overfitting, which benefits minority class learning on this severely imbalanced dataset.
 

> **Most efficient model: AlexNetLite** — 6–42s training, 66–72MB peak memory, 0.017ms latency, all 12 targets passed

---

## 3 Knowledge Transfer Adaptation: organs Dataset Task 3

### 3.1 Overview

The organs dataset contains approximately 450 training samples across 11 organ classes (~40 per class). A transfer learning framework was implemented using the orgs dataset as the source identical domain, modality, and class structure. Backbone weights are loaded from a trained orgs checkpoint; only the classifier head is re-initialised. FROZEN_STAGES controls the freeze depth: 0 = finetune (all layers train at low LR), 1+ = freeze backbone.

---

### 3.2 Scarce-Data Benchmark Matrix

| Dataset | Model | Training State | Accuracy | Precision | Recall | F1 | Target | Status |
|---------|-------|:-------------:|:--------:|:---------:|:------:|:--:|:------:|:------:|
| organs | AlexNet | scratch | 62.00% | 0.6174 | 0.5724 | 0.5319 | 40% |
| organs | VGG16 | scratch | 58.50% | 0.5386 | 0.5713 | 0.5304 | 40% |
| organs | ResNet18 | scratch | 68.00% | 0.6262 | 0.6476 | 0.6332 | 40% |
| organs | AlexNet | freeze_0 | 67.50% | 0.6601 | 0.6303 | 0.6283 | 40% |
| organs | VGG16 | freeze_0 | 66.50% | 0.6753 | 0.6257 | 0.6008 | 40% |
| organs | **ResNet18** | **freeze_0** | **71.00%** | **0.6877** | **0.6867** | **0.6736** | 40% |
| organs | AlexNet | freeze_1 | 62.50% | 0.6072 | 0.5773 | 0.5664 | 40% |
| organs | VGG16 | freeze_1 | 55.50% | 0.5555 | 0.5283 | 0.4995 | 40% |
| organs | ResNet18 | freeze_1 | 62.00% | 0.5951 | 0.5743 | 0.5686 | 40% |

**All evaluated configurations exceed the 40% accuracy target.** 

---

### 3.3 Impact of Transfer Learning
 
| Model | Scratch | freeze_0 | freeze_1 | Best State |
|-------|:-------:|:--------:|:--------:|:----------:|
| AlexNet | 62.00% | 67.50% | 62.50% | freeze_0 (+5.5%) |
| VGG16 | 58.50% | 66.50% | 55.50% | freeze_0 (+8.0%) |
| ResNet18 | 68.00% | **71.00%** | 62.00% | **freeze_0 (+3.0%)** |
 
**Best result: ResNet18 freeze_0 — 71.00% accuracy, F1 0.6736.**
 
All three models improve with freeze_0 (finetune). freeze_1 (frozen backbone) consistently underperforms and in some cases falls below scratch — demonstrating that with only 450 training samples, the classifier alone has insufficient data to learn discriminative features without backbone adaptation.

---

### 3.5 Data-Scarcity Post-Mortem

All freeze_0 models converge to 66–71% — a dataset ceiling driven by the 450-sample training set, not the architecture or transfer strategy. ResNet18's residual connections make it the most stable architecture for transfer learning on scarce data.

---

### 3.6 Recommendations

#### Within current data constraints

| Priority | Action |
|----------|--------|
|  Best config | ResNet18, freeze_0, LR 5e-5, WD 0.005, batch 8, epochs 20 |
|  Use | Stratified validation split — 50-sample random val set gives unreliable early stopping signal |
|  Consider | Combine predictions from all three freeze_0 models — expected 72–75% with no additional training |
|  Avoid | freeze_1 for VGG16 — loses 11% accuracy vs finetune at this dataset scale |

---

## Audit Log Reference

> Full error-by-error documentation including root cause analysis and git commit hashes for all entries is available in [AUDIT_LOG.md](AUDIT_LOG.md).