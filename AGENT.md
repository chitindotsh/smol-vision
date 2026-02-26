This file is the practical guide for agents working on this repository.
It is intentionally implementation-oriented: what to change, where, how to test,
and which behaviors are considered contractually stable.

## Project Scope

Pure C inference engine for SmolVLM-Instruct, a 2.2B parameter vision-language model:
- SigLIP-SO400M vision encoder (27 layers)
- Pixel shuffle connector (scale=3)
- SmolLM2-1.7B language model decoder (24 layers)

Primary target is CPU inference (BLAS + architecture-specific SIMD paths).

Based on the kernel and safetensors infrastructure from
[qwen-asr](https://github.com/antirez/qwen-asr) by Salvatore Sanfilippo.

## Source Of Truth

When docs and code disagree, trust these files first:
- CLI behavior and options: `smolvlm_main.c`
- Public API, config, and runtime state: `smolvlm.h`
- Model loading and generation orchestration: `smolvlm.c`
- Vision encoder math + load path: `smolvlm_vision.c`
- Decoder math + KV cache path: `smolvlm_decoder.c`
- Kernel dispatch and hot loops: `common_kernels*.c`, `common_kernels_impl.h`
- Image loading pipeline: `smolvlm_image.c`
- Tokenizer encode/decode: `smolvlm_tokenizer.c`
- Image loading tests: `test_smolvlm_images.c`
- Build targets: `Makefile`

Architecture/background references:
- `MODEL.md`

## User-Facing Behavior Contract (Do Not Break)

- `--silent` must still print generated text to stdout.
- `--silent` suppresses status/debug noise (stderr), not the text output.
- Without `--debug`, stderr should be concise:
  - model loading info
  - vision/connector progress
  - final inference summary line
- `--debug` enables verbose internal diagnostics (image dimensions, per-layer info).
- `-p` sets the text prompt (default: "Describe this image.").
- `-i` is required and accepts any image format supported by stb_image
  (PNG, JPG, BMP, PNM, TGA, GIF, PSD).
- `-d` is required and points to the model directory.
- `--max-tokens` controls maximum tokens generated (default: 256).

## Model + Inference Facts

- Model: SmolVLM-Instruct (HuggingFaceTB/SmolVLM-Instruct)
- Vision encoder: SigLIP-SO400M, 27 layers, 1152-dim, 16 heads, head_dim=72
  - Patch embedding: Conv2d with kernel=stride=14, padding=0
  - Learnable position embeddings (not sinusoidal)
  - Pre-LN transformer: LayerNorm(w/ bias) -> global attention(w/ bias) -> GELU FFN(w/ bias)
  - Post-layernorm after all layers
  - All vision weights are loaded as f32 (converted from bf16 at load where needed)
- Pixel shuffle connector: scale_factor=3, 729->81 tokens, dim 1152->10368
  - Single linear projection: Linear(10368->2048, no bias)
- Decoder: SmolLM2-1.7B (Llama-style)
  - 24 layers, 2048-dim, 32 heads (MHA, NOT GQA), head_dim=64
  - RoPE theta=273768.0, NeoX split-half style
  - RMSNorm eps=1e-5 (NOT 1e-6), NO per-head Q/K norms
  - SwiGLU MLP with fused gate+up interleaving
  - Separate lm_head (NOT tied to tok_embeddings)
  - Decoder large weights are bf16 mmapped and consumed via bf16 kernels
- Vocab: 49155 tokens, GPT-2 byte-level BPE from tokenizer.json
- Special tokens: image=49153, eos=49154, fake_image=49152, im_start=1

## Important Defaults

From `smolvlm_generate()` and CLI:
- Max tokens: 256
- Prompt: "Describe this image."
- Threads: all CPUs (via `qwen_get_num_cpus()`)
- Image size: 384x384 (from config.json, default)
- Patch size: 14 (from config.json, default)

## Repository Map

- `smolvlm_main.c`
  - CLI parsing, defaults, reporting, token streaming callback
- `smolvlm.c`
  - config.json parsing, model loading orchestration
  - `smolvlm_generate()`: image load -> vision encoder -> prompt build -> embedding merge -> prefill -> autoregressive decode
- `smolvlm_vision.c`
  - SigLIP vision encoder: patch embed, position embed, 27 transformer layers, post-LN
  - pixel shuffle: reshape 729->81 tokens with 9x dim expansion
  - connector linear projection: 10368->2048
- `smolvlm_decoder.c`
  - SmolLM2 decoder load + prefill + single-token forward + KV cache
  - fused gate+up weight interleaving at load time
  - argmax against separate lm_head (not tok_embeddings)
- `smolvlm_image.c`
  - stb_image-based loader (PNG/JPG/BMP/PNM/TGA/GIF/PSD)
  - bilinear resize to target_size x target_size
  - channel-first [3,H,W] normalization to [-1,1]
- `smolvlm_tokenizer.c`
  - GPT-2 byte-level BPE from tokenizer.json
  - encode text to token IDs, decode token IDs to text
- `common_kernels.c`
  - common math, threading, BLAS paths (shared from qwen-asr)
- `common_kernels_generic.c`
  - generic hot kernels
- `common_kernels_neon.c`
  - ARM NEON hot kernels
- `common_kernels_avx.c`
  - x86 AVX hot kernels
- `common_kernels_impl.h`
  - architecture dispatch macros
- `common_safetensors.c`
  - multi-shard safetensors loader with mmap
- `stb_image.h`
  - single-header image library (v2.30, public domain)
- `test_smolvlm_images.c`
  - image loading test suite (23 tests)
- `img_downloader.py`
  - test image downloader (picsum + synthetic patterns)
- `python_smolvlm.py`
  - Python reference implementation (standalone, numpy + safetensors + PIL)
- `download_model.sh`
  - model downloader (extensible multi-model pattern)

## Build + Run

Build:
```bash
make blas
```

Smoke run:
```bash
./smolvlm -d smolvlm-instruct -i test_images/building.jpg -p "What is in this image?"
```

Help:
```bash
./smolvlm -h
```

## Test Workflow

Image loading tests:
```bash
# Download test images first (if not present)
python3 img_downloader.py -o test_images

# Run tests (23 tests covering format variety, normalization, channel layout, resize)
make test-images
```

The test suite verifies:
- All supported formats load correctly at 384x384
- Pixel values are normalized to [-1, 1] range
- Output is channel-first [3, H, W] layout
- Solid color images produce correct per-channel values
- Resize works across sizes: 64, 128, 224, 384, 512

## Performance Reporting Contract

Final stderr summary line format is:
```text
Inference: <ms> ms, <tokens> tokens (<tok/s> tok/s, encoding: <ms>ms, decoding: <ms>ms)
```

`encoding` = image load + vision encoder + connector time
`decoding` = decoder prefill + autoregressive decode

## Kernel/Optimization Rules

- Architecture dispatch is centralized in `common_kernels_impl.h`.
- Keep generic/NEON/AVX variants functionally equivalent.
- If you optimize one path, verify no regression on others.
- Favor meaningful speedups; avoid complexity for tiny wins.
- The kernel files use the `common_` prefix (originally `qwen_asr_` from
  the qwen-asr project). Function names retain the `qwen_` prefix.

## Prompt Template

SmolVLM chat format:
```
<|im_start|>User:<fake_token_around_image><image>x81<fake_token_around_image>{prompt}<end_of_utterance>\nAssistant:
```

Token IDs:
```
[1] "User:" [49152] [49153 x 81] [49152] {prompt tokens} [49154] "\n" "Assistant:"
```

Where `<image>` positions (49153) are replaced with vision encoder output embeddings.
All other positions use text embeddings from `tok_embeddings_bf16`.

## Change Checklist For Agents

Before editing:
1. Identify behavioral contract impacted (CLI, output, speed, quality, memory).
2. Read corresponding source-of-truth file(s).

After editing:
1. Build: `make blas`
2. Run focused sanity command(s) for changed area.
3. Run image tests: `make test-images`
4. Update `README.md` if CLI/runtime behavior changed.
5. Keep `AGENT.md` aligned if workflow/test defaults changed.

## Local-Only Artifacts (Do Not Depend On In Commits)

Common local directories/files are intentionally ignored:
- `smolvlm-instruct/` (model weights)
- `test_images/` (downloaded test images)
- `TODO.md`
- virtualenv folders

Do not make code rely on these being present unless guarded by checks.
