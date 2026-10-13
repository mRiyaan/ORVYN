# NormWear Checkpoint Selection & Verification Analysis

## 1. Selected Checkpoint

- **Filename**: `normwear_pretrain_ckpt.pth`
- **Official Source**: GitHub Release `v1.0.0-alpha` (`Mobile-Sensing-and-UbiComp-Laboratory/NormWear`)
- **Download URL**: `https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear/releases/download/v1.0.0-alpha/normwear_pretrain_ckpt.pth`
- **Local Path**: `checkpoints/normwear_pretrain_ckpt.pth`
- **File Size**: `544,579,503` bytes (~519.36 MB)
- **SHA-256 Digest**: `36d0bca18356ccfc8e8916058bf838f26f1212a646f5780b487ad78581a92561`

---

## 2. Why It Is the Correct Foundation Checkpoint

1. **Official Release Documentation**:
   The release author (`yunfeiluo`) notes explicitly:
   > "`normwear_pretrain_ckpt.pth` is the latest checkpoint for the main backbone achieved during pretraining. This checkpoint is previously named `normwear_last_checkpoint-15470-correct.pth`. We update this checkpoint because we notice that the first release contain the optimizer state information which make the file unreasonably large. This latest version contain only the model weights."

2. **Downstream Feature Extraction Target**:
   In `NormWear/downstream_main.py` (lines 188-189):
   ```bash
   python3 -m NormWear.downstream_main --model_name normwear \
     --model_weight_dir data/results/normwear_last_checkpoint-15470-correct.pth
   ```
   This confirms that `normwear_last_checkpoint-15470-correct.pth` (now renamed `normwear_pretrain_ckpt.pth`) is the exact foundation model weight file used for all downstream evaluation tasks across the 14 benchmark datasets (including WESAD).

3. **Exact Key Matching**:
   When inspected with PyTorch, the checkpoint contains exactly 261 tensors corresponding 1-to-1 with the `NormWear` ViT-Base Masked Autoencoder backbone:
   - `cls_token` `[1, 1, 768]`
   - `pos_embed` `[1, 560, 768]`
   - `patch_embed.proj.weight` `[768, 3, 9, 5]`
   - 12 encoder blocks (`encoder_blocks.0` through `encoder_blocks.11`), each containing variate encoders and cross-channel fusion blocks
   - Decoder reconstruction weights (`decoder_embed`, `spatial_recon`, `temporal_recon`, `decoder_norm`)
   Loading into `NormWearModel` with default strict verification succeeds cleanly with 0 missing or unexpected keys.

---

## 3. Loading Method

The official repository loader in `NormWear.main_model.NormWearModel`:

```python
import torch
from NormWear.main_model import NormWearModel

ckpt_path = "checkpoints/normwear_pretrain_ckpt.pth"
model = NormWearModel(weight_path=ckpt_path, optimized_cwt=True)
model.eval()
```

Internal loading mechanism:
```python
stat_dict = torch.load(weight_path, map_location=torch.device('cpu'))
self.backbone.load_state_dict(stat_dict)
```
Executed strictly (`strict=True` by default in PyTorch), ensuring complete architectural alignment.

---

## 4. Evaluation of Alternative Checkpoints

### A. `normwear_msitf_zeroshot_last_checkpoint-5.pth` (GitHub Release)
- **File size**: 230,789,454 bytes (~220 MB)
- **Purpose**: Trained weights for the Memory Stream inspired Temporal Fusion (MSiTF) module used specifically in the zero-shot text-prompt alignment pipeline (`NormWear.zero_shot.msitf_fusion.NormWearZeroShot`).
- **Verdict: REJECTED for foundation embedding extraction.**
  This checkpoint contains only the text-fusion projection head, not the main physiological foundation backbone. It is meant to be loaded *alongside* the backbone for zero-shot text classification tasks, not as the primary signal encoder.

### B. `mosaic-laboratory/normwear` (`model.safetensors`, Hugging Face Hub)
- **File size**: 775 MB (193,812,170 parameters)
- **Purpose**: Hugging Face custom code packaging combining the backbone and MSiTF into a single `NormWearModel(PreTrainedModel)` wrapper.
- **Verdict: REJECTED for primary extraction in favor of official repository checkpoint.**
  - In `transformers >= 4.40 / 5.x`, `AutoModel.from_pretrained` initializes custom model classes inside a `torch.device("meta")` context manager.
  - The Hugging Face repo's `layers.py:608` contains a dummy probe:
    `return self.proj(torch.randn(1, 3, img_size[0], img_size[1]).to(device)).shape`
    This causes a fatal `NotImplementedError: Cannot copy out of meta tensor; no data!` during `AutoModel.from_pretrained`.
  - The GitHub release checkpoint `normwear_pretrain_ckpt.pth` pairs directly with the official repository code in `NormWear/main_model.py`, executes seamlessly on both CPU and Apple Silicon MPS, and is the ground-truth benchmark checkpoint evaluated in the paper.
