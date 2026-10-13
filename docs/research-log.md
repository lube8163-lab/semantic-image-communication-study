# Research log

## 2026-10-13 / Start authorized

The user approved starting the independent project after reviewing the scope. The original request is preserved at sources/project-request-ja.md. No existing Mim/iOS source is changed.

Read-only preflight: NVIDIA driver 581.80 recognizes RTX 5070 12GB and Tesla V100-PCIE-32GB. At the first sample the V100 had 9MiB used, 34C GPU and 33C HBM; RTX had about 847MiB used and 35C. CUDA's enumeration order differs from nvidia-smi, so use UUID rather than a hard-coded index. WSL is not installed. C: and D: have about 399GB and 688GB free respectively.

Existing PyTorch runtime was checked: Python 3.12.14, torch 2.10.0+cu128, sm_70 and sm_120 both in the build. It is used as a read-only dependency source. Additional packages go inside this project.

CoolingDesk reports fresh V100/HBM readings and active motherboard cooling, with no emergency. It also retains an RTX fan-watchdog expiry error; RTX manual fan control has returned to automatic (0 RPM at idle). This is not recorded as globally healthy cooling and is not repaired by this research task. Short V100 experiments will monitor temperature and freshness.

## Source findings before experiments

The official forward API already supports capture=(1-based block indices) and returns patch-token tensors. The trunk consumes image and evolving text tokens. The pixel head separately calls pixel_embedder(x), and the flow reconstruction computes x-sigma*v. Consequently a final token tensor alone is not the full state required by the unmodified existing head. Code locations: src/iris3b/models/dit.py and src/iris3b/sampling.py in the pinned upstream checkout.

Raw 256x256 tokens have shape [1,256,2560]: FP16 is 1,310,720 bytes versus 196,608 bytes for RGB8 (6.67x), before container overhead. This is architecture-derived arithmetic, not a measurement yet. At 1024x1024 the same ratio holds.

The paper's downstream depth/restoration comparison reports no significant pixel-space improvement. This is a different task from communication. It motivates avoiding an assumed VAE-free advantage.

## Initial checklist (completed; see results below)

- [x] Asset/model download and hashes.
- [x] Pretrained-model execution and finite-value check on V100.
- [x] Actual feature/packet bytes and receiver pilot.
- [x] Visual review, numerical summary, failure ledger and next-experiment decision.

## Initial milestone completed

Pretrained V100 execution: 15 forwards, 75 finite feature tensors, 30 independent receiver packets. Oracle head replay matched exactly for all 5 inputs. The frozen-head packet pilot did not improve fidelity over its thumbnail-only control and trailed byte-budget-matched JPEG/WebP. See initial-results-ja.md and results/pilot-20261013/report.html. Semantic superiority and a learned feature-conditioned decoder remain unverified.

## E004 completed: actual image -> text -> image comparison

Three frozen native VLMs produced78 raw captions on13 fixed inputs. Full independent Iris text sampling produced156 comparison images,2 empty controls and18 predefined same-seed40-step diagnostics. All176 image/packet pairs passed SHA and byte-count audit. Independent CLIP/DINO and separate pixel metrics cover459 reconstruction/control/codec observations. Main sampling uses native1024 on RTX5070 with BF16 and mask-preserving cuDNN SDPA,10 steps; native40 checks cover only the original three photos. Low-resolution256/512 failures and compatibility repairs are preserved and excluded from model-quality conclusions.

Qwen's detailed arm led both photo-average embedding metrics (CLIP.853/DINO.690, mean445B); its compact arm used265B and met the40-word target on all13 inputs. Comparisons are exploratory11-photo averages, not definitive general superiority. Readability/factual checks found spatial and object-detail omissions for all models. A matched-ceiling scaledWebP preserves pixels better while text generation preserves semantic resemblance better.

The native1024 feature follow-up used9 actual packets and18 decoded/thumbnail/zero/shuffle controls in separate sender/receiver processes. Oracle sender replay remained exact. Feature information affects the result, but pooled features plus thumbnail did not surpass thumbnail-only PSNR/DINO and cost about31.5kB. The current frozen-head protocol has no demonstrated communication-efficiency advantage; a learned compact adapter remains untested. See text-route-results-ja.md, text-route-protocol.md and results/text-route-20261013/report.html.
