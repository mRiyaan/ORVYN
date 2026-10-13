import sys
import os
import torch
import numpy as np

# Ensure repository is on python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from NormWear.main_model import NormWearModel

def run_smoke_test():
    ckpt_path = os.path.join(repo_root, "checkpoints", "normwear_pretrain_ckpt.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # Determine runtime device - per instructions: Try CPU first
    device_name = "CPU"
    device = torch.device("cpu")

    # Load actual model
    model = NormWearModel(weight_path=ckpt_path, optimized_cwt=True).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    commit_hash = "07517fcb13def8c89cb586128359cec02f86ec8d"

    runtime_info = f"""==========================================
REAL NORMWEAR FOUNDATION MODEL
==========================================

Model class:
{model.__class__.__module__}.{model.__class__.__name__} (Backbone: {model.backbone.__class__.__module__}.{model.backbone.__class__.__name__})

Checkpoint:
{ckpt_path}

Repository commit:
{commit_hash}

Python:
{sys.version.split()[0]}

PyTorch:
{torch.__version__}

Device:
{device_name}

Total parameters:
{total_params:,}

Trainable parameters:
{trainable_params:,}

=========================================="""

    print(runtime_info)

    # Smoke Test: Official input specification
    # Input format: [batch_size, n_channels, sequence_length]
    # Test tensor: batch_size=2, n_channels=3, 2 seconds of recording at sampling_rate=64 (128 samples)
    sampling_rate = 64
    batch_size = 2
    n_channels = 3
    seq_len = sampling_rate * 2  # 128 samples

    # Set seed for reproducible smoke test input
    torch.manual_seed(42)
    test_input = torch.rand(batch_size, n_channels, seq_len, dtype=torch.float32)

    # Run forward pass through official get_embedding interface
    with torch.no_grad():
        # Intermediate CWT computation
        cwt_spec = model.calc_cwt(test_input.numpy(), device=device) # [bn, nvar, 3, L, n_scales]
        # Backbone embedding extraction (with hidden_out to capture both final and layer outputs)
        encoder_raw, hidden_states = model.backbone.get_signal_embedding(
            cwt_spec.to(device), hidden_out=True, device=device
        )
        # Model embedding wrapper
        signal_embeddings = model.get_embedding(test_input, sampling_rate=sampling_rate, device=device)

    outputs_dict = {
        "cwt_scalogram": cwt_spec,
        "encoder_hidden_layers": len(hidden_states),
        "encoder_signal_embeddings": signal_embeddings,
        "cls_token_embedding": signal_embeddings[:, :, 0, :],
        "mean_pooled_embedding": signal_embeddings.mean(dim=2)
    }

    nan_count = int(torch.isnan(signal_embeddings).sum().item())
    inf_count = int(torch.isinf(signal_embeddings).sum().item())

    smoke_test_report = f"""==========================================
NORMWEAR SMOKE TEST
==========================================

Input shape:
{list(test_input.shape)}

Input dtype:
{test_input.dtype}

Output keys:
{list(outputs_dict.keys())}

For each output:
cwt_scalogram → {list(cwt_spec.shape)}
encoder_hidden_layers → {len(hidden_states)} layers (each {list(hidden_states[0].shape)})
encoder_signal_embeddings → {list(signal_embeddings.shape)}
cls_token_embedding → {list(outputs_dict['cls_token_embedding'].shape)}
mean_pooled_embedding → {list(outputs_dict['mean_pooled_embedding'].shape)}

Encoder output:
encoder_signal_embeddings

Encoder output shape:
{list(signal_embeddings.shape)}

NaN count:
{nan_count}

Inf count:
{inf_count}

=========================================="""

    print(smoke_test_report)

    # Numeric summary verification
    emb_mean = signal_embeddings.mean().item()
    emb_std = signal_embeddings.std().item()
    emb_min = signal_embeddings.min().item()
    emb_max = signal_embeddings.max().item()

    verification_report = f"""==========================================
EMBEDDING NUMERIC VERIFICATION
==========================================
Is not None: {signal_embeddings is not None}
Is Tensor: {torch.is_tensor(signal_embeddings)}
All values finite: {torch.isfinite(signal_embeddings).all().item()}
Mean: {emb_mean:.6f}
Std:  {emb_std:.6f}
Min:  {emb_min:.6f}
Max:  {emb_max:.6f}
=========================================="""

    print(verification_report)

    # Test MPS if available as secondary check per instructions
    mps_status = "Not attempted (CPU only)"
    if torch.backends.mps.is_available():
        try:
            model_mps = NormWearModel(weight_path=ckpt_path, optimized_cwt=True).to("mps")
            model_mps.eval()
            with torch.no_grad():
                out_mps = model_mps.get_embedding(test_input, sampling_rate=sampling_rate, device=torch.device("mps"))
            mps_status = f"SUCCESS - Output shape: {list(out_mps.shape)} (when CPU tensor input is passed)"
        except Exception as e:
            mps_status = f"FAILED - {e}"

    print(f"\nMPS Secondary Check: {mps_status}")

    # Save output report
    output_file = os.path.join(repo_root, "outputs", "normwear_smoke_test.txt")
    with open(output_file, "w") as f:
        f.write(runtime_info + "\n\n" + smoke_test_report + "\n\n" + verification_report + "\n\nMPS Secondary Check: " + mps_status + "\n")
    print(f"\nReport written to: {output_file}")

if __name__ == "__main__":
    run_smoke_test()
