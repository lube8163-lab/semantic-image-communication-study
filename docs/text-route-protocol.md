# E004: image -> factual text -> independently regenerated image

Exact parameters: configs/text-route.json. Original plan and abnormal outputs: results/text-route-20261013/diagnostic-256/. The input cohort and all raw captions were fixed before regeneration; no caption was rewritten after looking at outputs.

Candidates: Qwen3-VL-4B-Instruct (Apache-2.0; also Iris's required language encoder), MiniCPM-V4.6 (Apache-2.0), LFM2.5-VL-1.6B (LFM Open License v1.0, local research). Native Transformers without remote code, V100 FP16. Dependencies are project-local; shared models and Torch installation are read-only.

Inputs: three previous public photos and two synthetic controls, plus two deterministic hash-selected evaluation photos from each of four original keyword categories. Categories are metadata heuristics: selected objects include a car and selected landscapes include an archer. Verify source hashes and retain author/source/CC BY2.0 attribution. All senders see identical256px RGB center crops. Localized Narratives references are evaluator-only. This exploratory convenience cohort has unknown pretraining overlap and is not a benchmark holdout claim.

Each candidate emits English captions requested at most40/100 words, greedy decoding, repetition penalty1.05. Preserve raw outputs, actual lengths/UTF8 bytes/tokens, duration and failures. Do not edit, silently trim or discard conditions. Word targets are not exact byte-matched model arms. Wire bytes include complete JSON, resolution, seed and shared protocol id. Model distribution cost is excluded under the explicit assumption of pre-shared pretrained assets.

Receiver has text packets, shared models and public CPU RNG seeds only. Independently compute text-only Qwen states with the official Iris template and12 layers. No source pixels, thumbnails or image-conditioned VLM states enter it. Receiver-local states are recomputable and not transmitted. All78 captions fit Iris's295-token caption budget. Empty controls use the same independently encoded null.

## Resolution amendment before comparative batch

The original256px smoke produced binary stripes at both FP16 and FP32;512px remained visibly damaged. Native1024 produced coherent images at10/40 steps on V100/RTX. Exclude low-resolution diagnostics from VLM conclusions; preserve their original protocol, packets and images separately.

Main: Iris base at1024px, full rectified-flow DPM++ order2,10 steps, CFG3, shift4, seeds20261013/20261014. Three candidates x two lengths x thirteen sources x two seeds =156 images plus two empty controls. Save native images and downsample to256 with Lanczos for common-reference evaluation. This fast exploratory sampler is below the official100-step recommendation.

Predefined refinement: the three original photographs, all models/both lengths, first seed only,40 steps =18 images. Pair against main outputs at the exact same seed. Its shared protocol id explicitly differs and actual packet sizes are counted. Three photos cannot establish general ranking stability.

RTX5070 uses BF16 parameters/network and FP32 integrator. Windows Torch has no compiled FlashAttention. Select cuDNN SDPA only for the unmasked trunk/PiT and retain standard masked SDPA in the text refiner. No masks or equations change. V100 FP16 math-SDPA native diagnostics remain available. An initial all-cuDNN attempt rejected the text mask and is retained as a technical failure rather than bypassed.

## Controls and interpretation

Measure real JPEG/WebP files fitting each complete text-packet ceiling; report fixed256 failures. Also permit nine prespecified smaller resolutions decoded back to256; choose maximum source PSNR among a finite quality grid. This secondary diagnostic avoids confusing a256 encoding-size failure with inability to transmit a thumbnail. It does not establish rate-distortion optimality.

Repeat native1024 frozen-head checks on the three original photos because the initial feature pilot used256. Sender upscales the identical256 source, captures layer24 at sigma.1 and empty text, then transmits fullFP16 or pooled4x4/int4 features with/without a64 JPEG. A separate receiver reads only packets/weights/noise. Oracle replay with original noisy RGB is sender-only. Native thumbnail-only and zero/shuffled-feature decoder ablations identify side-information effects. Frozen head and text sampler use different compute/precision/byte budgets; this is not a learned feature codec comparison.

CLIP image cosine, DINOv2 image cosine, PSNR and MAE remain separate. DINO-family overlap with Iris's REPA teacher requires CLIP corroboration and visual checks. Review subjects, counts, distinct shape, colors and layout; high cosine does not certify identity or exact geometry. Average seeds within each photo before paired image bootstrap (10000 resamples, descriptive percentile95%, no multiplicity correction). Synthetic controls stay separate.

Sources: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct ; https://huggingface.co/openbmb/MiniCPM-V-4.6 ; https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B ; https://huggingface.co/speridlabs/iris-3b ; https://github.com/speridlabs/iris-3b .
