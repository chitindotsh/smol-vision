# smol-vision

Pure C inference engine for [SmolVLM-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct),
a 2.2B parameter vision-language model that combines a SigLIP vision encoder with a SmolLM2 language
model decoder.

Given an image and a text prompt, smol-vision generates a text description or answer — entirely in C,
with no external ML framework dependencies.

## Architecture

```
Image (PNG/JPG/BMP/PNM)
  -> Resize + Normalize
  -> SigLIP Vision Encoder (27 transformer layers)
  -> Pixel Shuffle (729 -> 81 tokens)
  -> Linear Connector
  -> Merge with text token embeddings
  -> SmolLM2 Decoder (24 transformer layers)
  -> Autoregressive text generation
```

| Component | Architecture |
|-----------|-------------|
| Vision encoder | SigLIP-SO400M — 1152-dim, 16 heads, 27 layers, GELU MLP, LayerNorm |
| Connector | Pixel shuffle (scale=3) + linear projection (10368 -> 2048) |
| Language model | SmolLM2-1.7B — 2048-dim, 32 heads (MHA), 24 layers, SwiGLU, RoPE, RMSNorm |
| Image formats | PNG, JPG, BMP, PNM, TGA, GIF, PSD (via [stb_image](https://github.com/nothings/stb)) |
| Weights | BF16 safetensors, memory-mapped (~4.5 GB) |
| Vocab | 49155 tokens, GPT-2 byte-level BPE |

## Quick Start

```bash
# Download model weights (~4.5 GB)
./download_model.sh

# Build
make blas

# Run
./smolvlm -d smolvlm-instruct -i photo.jpg -p "What do you see in this image?"
```

## Build

Requires GCC and a BLAS library (Apple Accelerate on macOS, OpenBLAS on Linux).

```bash
# Release build with BLAS
make blas

# Debug build with AddressSanitizer
make debug

# Show build configuration
make info
```

For Linux, install OpenBLAS first:
```bash
# Ubuntu/Debian
sudo apt install libopenblas-dev

# Fedora
sudo dnf install openblas-devel
```

## Usage

```
smolvlm -d <model_dir> -i <image> [options]

Required:
  -d <dir>          Model directory (with *.safetensors, tokenizer.json)
  -i <file>         Input image (PNG, JPG, BMP, PNM, TGA, GIF, PSD)

Options:
  -p <text>         Text prompt (default: "Describe this image.")
  -t <n>            Number of threads (default: all CPUs)
  --max-tokens <n>  Maximum tokens to generate (default: 256)
  --debug           Verbose debug output
  --silent          No status output (only generated text on stdout)
  -h                Show help
```

### Examples

```bash
# Describe an image
./smolvlm -d smolvlm-instruct -i photo.jpg

# Ask a specific question
./smolvlm -d smolvlm-instruct -i diagram.png -p "What does this diagram show?"

# Generate more text
./smolvlm -d smolvlm-instruct -i scene.jpg -p "Describe in detail" --max-tokens 512

# Silent mode (just the text, no status on stderr)
./smolvlm -d smolvlm-instruct -i photo.jpg --silent
```

## Testing

```bash
# Download test images (diverse photos + synthetic patterns)
python3 img_downloader.py

# Run image loading test suite (23 tests)
make test-images
```

## Project Structure

| File | Purpose |
|---|---|
| `smolvlm.h` | Public API, config/weight structs |
| `smolvlm.c` | Model loading, generation orchestration |
| `smolvlm_vision.c` | SigLIP vision encoder + pixel shuffle connector |
| `smolvlm_decoder.c` | SmolLM2 language model decoder |
| `smolvlm_image.c` | Image loading (stb_image), resize, normalize |
| `smolvlm_tokenizer.c` | BPE tokenizer (from tokenizer.json) |
| `smolvlm_main.c` | CLI entry point |
| `common_kernels*.c` | SIMD-optimized math kernels (generic/NEON/AVX) |
| `common_safetensors.c` | Safetensors loader with mmap |
| `python_smolvlm.py` | Python reference implementation |
| `img_downloader.py` | Test image downloader |
| `test_smolvlm_images.c` | Image loading test suite |

## Acknowledgments

This project is built on the kernel infrastructure and safetensors loader from
[qwen-asr](https://github.com/antirez/qwen-asr) by Salvatore Sanfilippo (antirez),
a pure C inference engine for Qwen3-ASR speech recognition models. The optimized
BLAS, NEON, and AVX kernel dispatch, threading, and model loading code are
reused directly.

## License

MIT License. See [LICENSE](LICENSE) for details.
