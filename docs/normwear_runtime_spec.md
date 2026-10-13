# NormWear Runtime Specification & Input/Output Interface

## 1. Input Specification

Based directly on the official implementation in `NormWear/main_model.py` and `NormWear/modules/normwear.py`:

### High-Level API Interface: `model.get_embedding(sample_data, sampling_rate, device)`

- **Data Container**: `torch.Tensor` or `numpy.ndarray`.
- **Dimensionality**: Exactly 3 dimensions: `[batch_size, nvar, sequence_length]`.
  - `batch_size` ($bn$): Number of windows/segments in the batch (e.g. 1, 2, 32, ...).
  - `nvar`: Number of sensor channels/variables (e.g., 3 for wrist PPG+EDA+TEMP, or 4 for default pretraining).
  - `sequence_length` ($L$): Number of time samples per window.
- **Data Type**: Float32 (`torch.float32` / `np.float32`).
- **Channel Permutations**: Channels are handled variate-wise. Channel order must be kept consistent across all samples and splits.
- **Sequence Length**:
  - For standard demonstration windows, 2 seconds @ 64 Hz ($L = 128$) produces 182 spatial/frequency patches + 1 `[CLS]` token = 183 tokens.
  - The model dynamically adjusts position embeddings via `pos_adjust((L, F_))` using bicubic interpolation on the 2D grid `(L // 9, F // 5)` so flexible sequence lengths are natively supported.
- **Sampling Rate Assumptions**:
  - The model's nominal target internal sampling rate is 65 Hz (or 64 Hz).
  - If input `sampling_rate > 256`, `get_embedding` automatically applies `torchaudio.transforms.Resample(sampling_rate, 65)`.
  - For sampling rates $\le 256$ (such as WESAD wrist sensors at 4 Hz, 32 Hz, or 64 Hz), signals should be properly resampled to 64 Hz / 65 Hz prior to batching.
- **Normalization Requirements**:
  - Signals should be zero-centered and scaled (e.g., using official `basic_preproc`: linear detrending, gaussian smoothing, followed by mean absolute amplitude normalization `ts /= np.mean(np.abs(ts))`).
- **CWT / Spectral Conversion**:
  - **Internal to `get_embedding`**: The user passes raw physiological time series `[bn, nvar, L]`. `NormWearModel.calc_cwt` computes the 3-channel Continuous Wavelet Transform scalograms internally using PyTorch-accelerated Ricker wavelets (`cwt_wrap`).
  - Scalogram output: `[bn, nvar, 3, L, 65]` where channel 0 is original signal, channel 1 is 1st difference $\Delta x$, and channel 2 is 2nd difference $\Delta^2 x$.

---

## 2. Model Backbone Specification

- **Architecture**: ViT-Base Masked Autoencoder with Cross-Variate CLS Fusion.
- **Patch Size**: $(9, 5)$ (9 time steps $\times$ 5 frequency scale bins).
- **Transformer Depth**: 12 layers.
- **Attention Heads**: 12 heads.
- **Embedding Dimension ($E$)**: 768.
- **Decoder Depth**: 2 layers (hidden size: 512).
- **Fusion Frequency**: Cross-channel `[CLS]` token fusion executed every 2 layers (`fuse_frequency = 2`).

---

## 3. Output Specification

### Raw Encoder Output:
- **Output Tensor**: `signal_embeddings`
- **Output Shape**: `[batch_size, nvar, P, 768]`
  - `batch_size`: Batch dimension.
  - `nvar`: Sensor channel dimension.
  - `P`: Number of patches + 1 (the 0th index along this axis is the `[CLS]` token).
  - `768`: Representation hidden dimension.

### Layer Hidden States (via `hidden_out=True`):
- List of 12 intermediate tensors, each of shape `[batch_size, nvar, P, 768]`.

### Official Aggregation Strategies:
1. **Patch Pooling**:
   - `[CLS]` Token Representation:
     ```python
     cls_embed = out[:, :, 0, :]  # Shape: [batch_size, nvar, 768]
     ```
   - Mean Patch Pooling:
     ```python
     mean_embed = out.mean(dim=2)  # Shape: [batch_size, nvar, 768]
     ```
2. **Channel Pooling**:
   - Channel Concatenation:
     ```python
     flattened_embed = cls_embed.flatten(start_dim=1)  # Shape: [batch_size, nvar * 768]
     ```
   - Channel Mean Pooling:
     ```python
     pooled_embed = cls_embed.mean(dim=1)  # Shape: [batch_size, 768]
     ```

---

## 4. Hardware and Runtime Behavior

- **CPU**: Standard native execution via `torch.device('cpu')`.
- **Apple Silicon MPS**: Fully operational when tensors are input from host CPU memory and moved to MPS for forward execution (`device=torch.device('mps')`).
- **Memory Footprint**: ~545 MB parameters on disk; ~1.2 GB active RAM during inference.
- **Inference Mode**: Always evaluate with `model.eval()` and `torch.no_grad()`.
