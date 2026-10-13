# 初期実験設計（実行前）

作成日: 2026-10-13

## Question and hypotheses

- H1: trained intermediate patch tokens contain input-specific visual information that changes with layer and noise level.
- H2: raw tokens are not automatically a useful compressed representation; token/channel reduction and quantization need measured byte accounting.
- H3: the existing pixel head requires the noisy RGB input as well as final patch tokens, so a feature-only receiver may fail even if the features are informative.

These are hypotheses, not results.

## E001: pretrained feature observation

Use the base Iris-3B checkpoint, the official empty-prompt states, three public photographs and two controlled synthetic scenes, cropped/resized to RGB 256x256. Fixed sigma values 0.1, 0.5 and 0.9, seed 20261013. Define x_sigma=(1-sigma)*x0+sigma*epsilon. Capture blocks 1, 8, 10, 16 and 24 (1-based). Record feature shape, finite values, uncompressed/container sizes, runtime and GPU memory. Single-step x0 estimates are denoising diagnostics and are not a feature codec.

## E002: frozen final-head packet pilot

Use block 24 tokens from sigma 0.1. Compare full tokens, spatially pooled 4x4 tokens, int8 and packed int4 quantization. The receiver gets a serialized packet, shared checkpoint, shared PRNG algorithm and an optional 64x64 JPEG thumbnail. It must not read the original image or encoder tensors. Without a thumbnail it uses noise-only pixel input. With a thumbnail it reconstructs the approximate noisy input from the transmitted thumbnail and seed. The final PiT head is frozen and no adapter is trained.

Also compare an oracle head replay using exact original noisy RGB, explicitly outside the communication results. This validates the extraction/head split and detects hidden bypass information. Quantization and pooling are a deliberately simple intervention, not a trained bottleneck.

All packet bytes count, including array/container headers, scale tables, thumbnail, seed, shape, sigma and model revision. Model weights and fixed empty prompt are shared assets; record their size separately. Per-image names and ground-truth hashes belong in evaluation manifests, not decoder conditioning.

## Controls and evaluation

At identical 256x256 output compare thumbnail-only, JPEG and WebP at bytes <= the tested packet. Select codec quality by byte budget, without quality-score selection. Save originals and every compared output. PSNR/MAE measure fidelity only. No independent semantic metric is yet selected, so do not infer semantic superiority from PSNR or Iris self-similarity. Three photographs and two synthetic images establish operational feasibility only, not generalization.

A later independent semantic evaluator / held-out set, a learned bottleneck, earlier-layer adapter or different model should be chosen only after these results.

## Environment and stop conditions

Prefer V100 UUID <gpu-id-redacted>, PyTorch 2.10.0+cu128 with sm_70. Try FP32 parameters with FP16 autocast and math SDPA, then use FP32 if non-finite. Treat that precision change as a recorded deviation from the official BF16 path. Keep batch=1 and resolution=256. Cooling settings remain owned by CoolingDesk; record fresh V100/HBM and board status. Stop before a pass if GPU or HBM reaches 70C, monitoring is stale, or board cooling reports an emergency.

## References

- Iris-3B paper: https://arxiv.org/abs/2610.09450 ; official source revision cee0c130a62a9c288ad8918f1f7cdf23cc228d69.
- REPA: https://arxiv.org/abs/2410.06940 . Alignment helps representation learning; it does not define an image decoder.
- Diffusion Autoencoders: https://arxiv.org/abs/2111.15640 . Explicitly separates semantic and stochastic information and learns a decoder; a useful contrast to feature extraction alone.
- Diffusion-Aided Bandwidth-Efficient Semantic Communication with Adaptive Requests: https://arxiv.org/abs/2510.26442 . Text plus visual side information is a relevant comparison concept; this pilot does not reproduce that system.
- Public image/license documentation: https://scikit-image.org/docs/stable/api/skimage.data.html .
