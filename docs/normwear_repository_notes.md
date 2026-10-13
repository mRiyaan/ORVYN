# NormWear Repository Notes & Technical Audit

## 1. Repository Overview & Metadata

- **Repository Name**: NormWear
- **URL**: `https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear`
- **Commit Hash**: `07517fcb13def8c89cb586128359cec02f86ec8d`
- **Branch**: `main`
- **Local Path**: `NormWear/` (relative to workspace root `/Users/mohammedriyaan/capstone-model-demo`)
- **License**: Apache License 2.0
- **Associated Publication**: "Toward Foundation Model for Multivariate Wearable Sensing of Physiological Signals" (Luo et al., ACM Transactions on Computing for Healthcare, March 2026 / arXiv:2412.09758).
- **Hugging Face Repository**: `https://huggingface.co/mosaic-laboratory/normwear` (Custom code: `NormWearConfig`, `NormWearModel`, `model.safetensors`, `pytorch_model.bin`)
- **Official GitHub Release**: `https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear/releases/tag/v1.0.0-alpha`

---

## 2. Model Implementation Architecture

The core architecture is implemented across `NormWear/main_model.py` and `NormWear/modules/`:

### Primary Classes
1. **`NormWearModel` (`NormWear/main_model.py`)**:
   - High-level wrapper around the `NormWear` backbone.
   - Initializer arguments:
     - `optimized_cwt` (bool, default `False`): Uses PyTorch-accelerated Continuous Wavelet Transform (`cwt_ricker`/`cwt_wrap`) instead of SciPy's `signal.cwt`.
     - `use_cwt` (bool, default `True`): Whether to compute scalograms or operate directly on raw 1D time series.
     - `weight_path` (str, default `""`): Path to PyTorch `.pth` checkpoint.
   - Sampling rate expectation: Default internal target sampling rate is 65 Hz (resamples if input `sampling_rate > 256` via `torchaudio.transforms.Resample`).

2. **`NormWear` Backbone (`NormWear/modules/normwear.py`)**:
   - Masked Autoencoder (MAE) with a Vision Transformer (ViT) backbone adapted for multi-variate wearable signals.
   - Default hyperparameters:
     - `img_size=(387, 65)`
     - `patch_size=(9, 5)`
     - `in_chans=3` (3 scalogram channels: signal $x$, first difference $\Delta x$, second difference $\Delta^2 x$)
     - `embed_dim=768` (Standard ViT-Base hidden size)
     - `depth=12` transformer layers
     - `num_heads=12` attention heads
     - `decoder_embed_dim=512`
     - `decoder_depth=2`
     - `nvar=4` default variable capacity (can adapt dynamically)
     - `fuse_freq=2` (cross-channel [CLS] fusion every 2 layers via `EncoderLayer`)

3. **`PatchEmbed_new` (`NormWear/modules/patch_embed.py`)**:
   - 2D Conv projection mapping scalogram patches `(9, 5)` into `768` dimensions.

4. **`EncoderLayer` (`NormWear/modules/normwear.py`)**:
   - Applies variate encoder (`Block` from `timm`) per sensor channel.
   - Every `fuse_frequency` layers (default 2), executes `cls_fusion` across channels on the `[CLS]` token.

---

## 3. Preprocessing Functions

Located in `NormWear/modules/signal_preprocess.py` and `NormWear/main_model.py`:

1. **`basic_preproc`**:
   - Outlier detection: Calculates first difference $\Delta ts$ and clips using quantile threshold (`outlier_p=0.95`).
   - Imputation: Linear interpolation (`impute`) over missing/NaN samples.
   - Resampling: Interp-based downsampling from source `sr` (e.g. 1000 Hz) to target `tr` (e.g. 65 Hz / 125 Hz).
   - Detrending: Linear detrending via `scipy.signal.detrend`.
   - Smoothing: 1D Gaussian smoothing (`scipy.ndimage.gaussian_filter`) with `sigma = tr * 0.02`.
   - Normalization: Scale by mean absolute amplitude: `ts /= np.mean(np.abs(ts))`.

2. **`preproc_all(all_tss, ss, ts=65, ...)`**:
   - Batches `basic_preproc` across all $C$ channels of shape `(C, L)`.

3. **Continuous Wavelet Transform (CWT)**:
   - Input: Raw physiological signal windows.
   - Computes:
     - $x$ (signal)
     - $x_1 = x[1:] - x[:-1]$ (first difference)
     - $x_2 = x_1[1:] - x_1[:-1]$ (second difference)
   - Stacks Ricker wavelet scalograms into 3 channels `(3, L, F)` per sensor variable.
   - Two modes provided:
     - Vanilla: `scipy.signal.cwt(ts, signal.ricker, np.arange(0.1, 65))`
     - Optimized: PyTorch GPU/CPU native 1D convolution with Ricker wavelet kernels (`cwt_ricker` in `normwear.py`).

---

## 4. Model Loading & Checkpoint Instructions

Two official paths exist to obtain and load NormWear checkpoints:

### Path A: GitHub Release Checkpoint (`v1.0.0-alpha`)
- **Release tag**: `v1.0.0-alpha`
- **Release Assets**:
  - `normwear_pretrain_ckpt.pth` (Size: 544,579,503 bytes, SHA256: `36d0bca18356ccfc8e8916058bf838f26f1212a646f5780b487ad78581a92561`): Official backbone weights without training optimizer states.
  - `normwear_msitf_zeroshot_last_checkpoint-5.pth` (Size: 230,789,454 bytes): Checkpoint for the Memory Stream inspired Temporal Fusion (MSiTF) module for zero-shot text-signal alignment.
- **Loading code**:
  ```python
  from NormWear.main_model import NormWearModel
  model = NormWearModel(weight_path="path/to/normwear_pretrain_ckpt.pth", optimized_cwt=True)
  model.eval()
  ```

### Path B: Hugging Face Hub (`mosaic-laboratory/normwear`)
- **Hugging Face Model ID**: `mosaic-laboratory/normwear`
- **Files**: `config.json`, `configuration_normwear.py`, `modeling_normwear.py`, `model.safetensors`, `pytorch_model.bin`.
- **Loading code**:
  ```python
  from transformers import AutoModel
  model = AutoModel.from_pretrained("mosaic-laboratory/normwear", trust_remote_code=True)
  model.eval()
  ```

---

## 5. Example Inference Code

From official `README.md` and `main_model.py`:

```python
import torch
from NormWear.main_model import NormWearModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NormWearModel(weight_path="path/to/checkpoint.pth", optimized_cwt=True).to(device)
model.eval()

# Input format: [batch_size, n_channels, sequence_length]
# Example: 2 samples, 3 sensor channels, 2 seconds at 64 Hz (128 time points)
sampling_rate = 64
x = torch.rand(2, 3, sampling_rate * 2).to(device)

with torch.no_grad():
    out = model.get_embedding(x, sampling_rate=sampling_rate, device=device)
    # out shape: [batch_size, n_channels, num_patches, 768]

# Pooling options across patches:
# [CLS] token: out[:, :, 0, :]               -> [batch_size, n_channels, 768]
# Mean pool:   out.mean(dim=2)              -> [batch_size, n_channels, 768]

# Pooling options across sensor channels:
# Concat:      embedding.flatten(start_dim=1) -> [batch_size, n_channels * 768]
# Mean:        embedding.mean(dim=1)        -> [batch_size, 768]
```

---

## 6. Dependencies & Requirements

- Python >= 3.7 (officially recommended: Python 3.11)
- `torch`, `torchvision`, `torchaudio`
- `transformers`, `huggingface_hub`
- `timm` (PyTorch Image Models, needed for transformer `Block`)
- `scipy`, `numpy`, `pandas`, `scikit-learn`
- `tabulate`, `tqdm`
- Baseline requirements (optional): `chronos`, `pytorch_pretrained_vit`

---

## 7. Downstream & Fine-Tuning Pipelines

- **Downstream Orchestrator**: `NormWear/downstream_main.py`
  - Evaluates on 14 datasets, including `wearable_downstream/wesad`.
- **Feature Extraction & Saving**: `NormWear/downstream_pipeline/prepare_embeds.py`
  - Calls `NormWear_API.get_embedding` and caches embeddings per subject and window.
- **Linear Probing**: `NormWear/downstream_pipeline/linear_prob_main.py`
- **End-to-End Fine-Tuning**: `NormWear/downstream_pipeline/main_finetune.py`
- **Dataset Specification Format**:
  - Sample pickle files: `{'uid': subject_id, 'data': np.array([num_channels, sequence_length]), 'sampling_rate': fs, 'label': [{'class': c}, ...]}`
  - Splits: `train_test_split.json` containing subject/sample partition lists.
