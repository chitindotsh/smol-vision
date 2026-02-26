# SmolVLM-Instruct — Model Reference

Model: `HuggingFaceTB/SmolVLM-Instruct` (2.2B parameters)

This document describes the model architecture, weight format, tokenizer layout,
and inference algorithm needed to implement SmolVLM-Instruct from scratch.
The Python reference implementation (`python_smolvlm.py`) is the
executable version of this document.

---

## Architecture Overview

SmolVLM-Instruct is a vision-language model based on Idefics3ForConditionalGeneration.
It has three main components:
- **Vision Encoder (SigLIP-SO400M)**: Patch embedding + transformer encoder
- **Connector**: Pixel shuffle token compression + linear projection
- **Language Model Decoder (SmolLM2-1.7B)**: Llama-style transformer with SwiGLU

**Pipeline:**
```
Image → Resize 384×384 → Normalize [-1,1] → Patch Embed (Conv2d 14×14)
  → 27 SigLIP Transformer Layers → Post-LayerNorm
  → Pixel Shuffle (729→81 tokens) → Linear Projection (10368→2048)
  → Merge with text token embeddings
  → 24 SmolLM2 Decoder Layers → Final RMSNorm → LM Head → Tokens
```

### Model Parameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| **Vision** | d_model | 1152 |
| | n_layers | 27 |
| | n_heads | 16 |
| | head_dim | 72 |
| | FFN dim | 4304 |
| | image_size | 384 |
| | patch_size | 14 |
| | num_patches | 729 (27×27) |
| | norm | LayerNorm (eps=1e-6, with bias) |
| | activation | GELU (tanh approximation) |
| **Connector** | scale_factor | 3 |
| | output tokens | 81 (729/9) |
| | shuffle dim | 10368 (1152×9) |
| | projection | Linear(10368→2048, no bias) |
| **Decoder** | hidden_size | 2048 |
| | n_layers | 24 |
| | n_heads | 32 (MHA) |
| | n_kv_heads | 32 (MHA, NOT GQA) |
| | head_dim | 64 |
| | intermediate_size | 8192 |
| | norm | RMSNorm (eps=1e-5) |
| | position | RoPE (theta=273768, NeoX style) |
| | vocab_size | 49155 |
| | tied embeddings | NO (separate lm_head) |

---

## Image Preprocessing

| Parameter | Value |
|-----------|-------|
| Input formats | PNG, JPG, BMP, PNM, TGA, GIF, PSD (via stb_image) |
| Target size | 384×384 pixels |
| Resize method | Bilinear interpolation |
| Normalization | pixel/255 * 2 - 1 (range [-1, 1]) |
| Channel order | RGB, channel-first [3, H, W] |

**Steps:**
1. Load image as RGB uint8 (any format, forced to 3 channels)
2. Bilinear resize to 384×384
3. Convert to float, normalize each pixel: `val = pixel/255.0 * 2.0 - 1.0`
4. Rearrange from row-major interleaved `[H, W, 3]` to channel-first `[3, H, W]`

Note: SmolVLM's official preprocessor uses `image_mean=[0.5, 0.5, 0.5]` and
`image_std=[0.5, 0.5, 0.5]`, which is equivalent to `(pixel/255 - 0.5) / 0.5 = pixel/255 * 2 - 1`.

---

## SigLIP Vision Encoder

SigLIP (Sigmoid Language-Image Pre-training) is a vision transformer trained with
a sigmoid loss instead of the traditional softmax contrastive loss used in CLIP.
The key insight is that sigmoid loss operates on individual image-text pairs rather
than requiring pairwise comparisons across a batch, enabling better scaling to
larger batch sizes and more efficient training.

SmolVLM uses the SigLIP-SO400M variant — a ViT-SO400M architecture (Shape-Optimized
at 400M parameters) pretrained on image-text pairs.

### SigLIP in the SmolVLM Pipeline

In SmolVLM, SigLIP functions purely as a visual feature extractor. Its contrastive
text tower is discarded; only the vision transformer is retained. The encoder
converts a 384×384 image into a sequence of 729 patch embeddings (27×27 spatial grid),
which are then compressed via pixel shuffle and projected into the language model's
embedding space via a linear connector.

SigLIP was chosen for SmolVLM over standard CLIP encoders because:
- The sigmoid loss produces more calibrated per-image features (no batch dependency)
- The SO400M architecture is shape-optimized for the 384×384 input resolution
- It provides strong visual representations while remaining compact enough for
  a 2.2B total parameter budget

### Patch Embedding

The first operation converts the raw image into a sequence of patch tokens:

```
Conv2d(in_channels=3, out_channels=1152, kernel_size=14, stride=14, padding=0)
```

Input: `[3, 384, 384]` (channel-first RGB image, normalized to [-1,1])
Output: `[1152, 27, 27]` → reshape to `[729, 1152]` (num_patches × hidden_dim)

This is equivalent to splitting the image into a 27×27 grid of 14×14 pixel
patches, and projecting each patch into a 1152-dimensional embedding via a
learned linear transformation.

Unlike standard ViTs which prepend a `[CLS]` token, SigLIP does not use one.
All 729 patch tokens participate equally in the transformer layers.

A learnable bias is added to each patch embedding after the convolution.

### Position Embeddings

After patch embedding, **learnable** position embeddings are added:

```python
output = patch_embeddings + position_embedding.weight[:num_patches]
```

Shape: `[num_positions, 1152]` where `num_positions >= 729`.

These are NOT sinusoidal (as in the original ViT paper or Qwen3-ASR's encoder).
They are fully learned parameters stored in the model weights. Each spatial
position in the 27×27 grid has its own learned embedding vector.

### Transformer Layers (×27)

Each of the 27 SigLIP encoder layers follows a Pre-LN transformer pattern:

```
residual = h
h_norm = LayerNorm(h, ln1_weight, ln1_bias, eps=1e-6)

q = h_norm @ Wq + bq    # [729, 1152]
k = h_norm @ Wk + bk    # [729, 1152]
v = h_norm @ Wv + bv    # [729, 1152]
attn_out = global_bidirectional_attention(q, k, v)
h = residual + (attn_out @ Wo + bo)

residual = h
h_norm = LayerNorm(h, ln2_weight, ln2_bias, eps=1e-6)
ffn_out = GELU(h_norm @ W_fc1 + b_fc1) @ W_fc2 + b_fc2
h = residual + ffn_out
```

**Key characteristics:**
- **LayerNorm with bias** (NOT RMSNorm): Both weight and bias parameters.
  This differs from the decoder which uses RMSNorm without bias.
- **All linear layers have biases**: Q, K, V, output projection, FC1, FC2
  all include bias terms. This is a major difference from the decoder where
  no layer has biases.
- **Global bidirectional attention**: Every patch can attend to every other
  patch. There is NO causal mask, NO windowed attention, and NO local
  attention patterns. The full 729×729 attention matrix is computed.
- **GELU activation** (tanh approximation): `GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.
  This is the same variant used in GPT-2 and BERT.

**Attention computation:**
```python
scale = 1.0 / sqrt(head_dim)   # = 1/sqrt(72) ≈ 0.1179
q, k, v = reshape to [n_heads=16, seq=729, head_dim=72]
attn_weights = q @ k.T * scale  # [16, 729, 729]
# NO mask applied (bidirectional)
attn_weights = softmax(attn_weights, dim=-1)
attn_output = attn_weights @ v
attn_output = reshape to [729, 1152]
```

### Post-LayerNorm

After all 27 transformer layers, a final LayerNorm is applied:

```python
output = LayerNorm(h, post_layernorm_weight, post_layernorm_bias, eps=1e-6)
```

This is the last operation in the vision encoder before the connector.

### Key Differences: SigLIP vs Qwen3-ASR Encoder

| Feature | SigLIP (SmolVLM) | Qwen3-ASR Encoder |
|---------|------------------|-------------------|
| Input | Image patches (14×14) | Mel spectrogram chunks |
| Stem | Conv2d (single layer, k=14 s=14) | Conv2d ×3 (8× time downsample) |
| Position embeddings | Learnable (stored in weights) | Sinusoidal (computed at runtime) |
| Attention | Global bidirectional (full 729×729) | Windowed bidirectional (~104 tokens) |
| Norm | LayerNorm with bias | LayerNorm with bias |
| Biases | YES on all projections | YES on all projections |
| Output | 729 patch tokens (no CLS) | Variable-length token sequence |
| Post-processing | Pixel shuffle + linear | GELU MLP projection (2 layers) |

Both encoders share the kernel functions: `qwen_layer_norm`, `qwen_linear`,
`qwen_bidirectional_attention`, `qwen_gelu`, and `qwen_conv2d`.

---

## Pixel Shuffle Connector

The pixel shuffle connector compresses the 729 vision tokens into 81 tokens,
while expanding the feature dimension. This is a spatial compression technique
that trades sequence length for feature richness.

### Pixel Shuffle Operation

The operation groups `scale_factor × scale_factor = 3×3 = 9` spatially adjacent
patch tokens into a single token by concatenating their features:

```
Input:  [729, 1152]  → view as [27, 27, 1152] spatial grid
Output: [81, 10368]  → view as [9, 9, 10368] spatial grid
```

**Exact PyTorch implementation** (from SmolVLM source):
```python
x = x.view(h, w, dim)               # [27, 27, 1152]
x = x.view(h, w//sf, dim*sf)        # [27, 9, 3456]   — group along w
x = x.transpose(0, 1)               # [9, 27, 3456]
x = x.reshape(w//sf, h//sf, dim*sf*sf)  # [9, 9, 10368]  — group along h
x = x.transpose(0, 1)               # [9, 9, 10368]
x = x.reshape(new_seq, new_dim)     # [81, 10368]
```

**C implementation** uses a direct index mapping. For each output position
`(oh, ow)`, the source is gathered from a 3×3 spatial block with the transposed
index pattern `(oh*sf + sw, ow*sf + sh)` — note the swapped `sh`/`sw` due to
the two transposes in the PyTorch sequence.

### Linear Projection

After pixel shuffle, a single linear projection maps the expanded features
to the decoder's hidden dimension:

```
Linear(10368 → 2048, no bias)
```

This is the only trainable component in the connector. Unlike the original
Idefics2 connector which used a 2-layer MLP with GELU, SmolVLM uses a single
linear layer.

Tensor name: `model.connector.modality_projection.proj.weight` (shape [2048, 10368])

---

## LLM Decoder (SmolLM2-1.7B)

SmolLM2 is a Llama-style transformer. It differs from the Qwen3 decoder used
in Qwen3-ASR in several significant ways.

### Key Differences from Qwen3

| Feature | SmolLM2 (SmolVLM) | Qwen3 (Qwen3-ASR) |
|---------|--------------------|--------------------|
| Attention | MHA (32 Q = 32 KV) | GQA (16 Q / 8 KV) |
| head_dim | 64 | 128 |
| Per-head Q/K RMSNorm | NO | YES |
| RoPE theta | 273768.0 | 1,000,000.0 |
| RMSNorm eps | 1e-5 | 1e-6 |
| lm_head | Separate weights | Tied to tok_embeddings |
| Biases | None | None |

### Decoder Forward Pass

Per-layer computation for hidden state `h` at positions `pos..pos+seq-1`:

1. **Input RMSNorm**: `x = RMSNorm(h, input_layernorm, eps=1e-5)`
2. **QKV projections (MHA)**:
   - `q = x @ Wq^T` → `[seq, 32×64]` → reshape `[seq, 32, 64]`
   - `k = x @ Wk^T` → `[seq, 32×64]` → reshape `[seq, 32, 64]`
   - `v = x @ Wv^T` → `[seq, 32×64]`
3. **NO per-head Q/K RMSNorm** (unlike Qwen3, SmolLM2 skips this entirely)
4. **RoPE** on Q and K (NeoX split-half style, theta=273768)
5. **KV cache**: append K, V to per-layer cache
6. **Causal attention**: scale=1/sqrt(64), MHA (no GQA repeat needed)
7. **Output projection + residual**: `h = h + attn_out @ Wo^T`
8. **Post-attention RMSNorm**: `h_norm = RMSNorm(h, post_attention_layernorm, eps=1e-5)`
9. **SwiGLU MLP + residual**:
   - `gate = silu(h_norm @ W_gate^T)`
   - `up = h_norm @ W_up^T`
   - `h = h + (gate * up) @ W_down^T`

After last layer: `h = RMSNorm(h, norm.weight)`, then `logits = h @ lm_head^T`.

### RoPE (NeoX/split-half style)

```python
inv_freq = 1.0 / (theta ** (arange(0, head_dim, 2) / head_dim))  # [32]
angles = positions * inv_freq    # [seq, 32]
emb = cat(angles, angles)        # [seq, 64] (duplicate for full head_dim)
cos, sin = emb.cos(), emb.sin()

# rotate_half: x1 = x[..., :32], x2 = x[..., 32:]
# result = x * cos + cat(-x2, x1) * sin
```

Where theta=273768.0 (a notably unusual value, likely tuned for SmolLM2's
specific training distribution).

### SwiGLU MLP

The MLP uses the SwiGLU activation pattern:
```python
gate = silu(x @ W_gate^T)   # [seq, 8192]
up   = x @ W_up^T           # [seq, 8192]
out  = (gate * up) @ W_down^T  # [seq, 2048]
```

For single-token forward pass, gate and up weights are fused into an
interleaved layout `[gate_row0, up_row0, gate_row1, up_row1, ...]` for
efficient matvec dispatch.

### LM Head (Separate)

Unlike Qwen3 where `lm_head` is tied to `embed_tokens`, SmolLM2 uses a
**separate** lm_head weight matrix:

```
embed_tokens: [49155, 2048]  — for converting token IDs to embeddings (input)
lm_head:      [49155, 2048]  — for projecting hidden states to logits (output)
```

These are independent parameters. During generation, argmax is computed
against `lm_head`, not `embed_tokens`.

---

## Tokenizer (GPT-2 BPE)

### Special Token IDs

```
<|im_start|>                 = 1       (also BOS)
<|im_end|>                   = 2       (also pad)
<fake_token_around_image>    = 49152
<image>                      = 49153   (placeholder for vision embeddings)
<end_of_utterance>           = 49154   (EOS for generation)
```

EOS token IDs: `{49154}`

### Token Encoding/Decoding

Uses GPT-2 style byte-level BPE loaded from `tokenizer.json` (HuggingFace
fast tokenizer format). The tokenizer.json contains both the vocabulary
(model.vocab dict) and merge rules (model.merges array), plus added_tokens
for special tokens.

Characters are encoded using the GPT-2 bytes-to-unicode mapping (printable
ASCII + extended Latin-1, with remaining bytes mapped to Unicode chars
starting at U+0100).

To decode: look up token string in inverted vocab → convert each character
through reverse byte mapping → decode resulting bytes as UTF-8.

---

## Prompt Format

The prompt template for SmolVLM vision-language inference:

```
<|im_start|>User:<fake_token_around_image><image>×81<fake_token_around_image>{prompt}<end_of_utterance>\nAssistant:
```

As token IDs:
```
PREFIX:  [1]  (im_start)
USER:    tokenize("User:")
FAKE1:   [49152]
IMAGE:   [49153] × 81  (image_seq_len)
FAKE2:   [49152]
PROMPT:  tokenize(prompt_text)
EOS:     [49154]
NEWLINE: tokenize("\n")
ASST:    tokenize("Assistant:")
```

Where the 81 `<image>` token positions (49153) are replaced with vision
encoder output embeddings during embedding construction. All other positions
use text embeddings from `tok_embeddings_bf16`.

---

## Embedding Merge Strategy

SmolVLM uses a **replacement** strategy (same as Qwen3-ASR for audio):

1. **Build prompt**: Construct input_ids with text tokens + `<image>×81` placeholders
2. **Embed tokens**: Look up all token embeddings via `tok_embeddings_bf16`
3. **Replace image positions**: Find positions where `input_ids == 49153` and
   replace those embeddings with the corresponding vision encoder outputs
4. **Prefill**: Feed combined embedding sequence through decoder (all-but-last)
5. **First token**: Run decoder forward on last prompt embedding
6. **Autoregressive decode**: For each subsequent step, embed the previous token,
   feed through decoder, greedy argmax. Stop on EOS (49154) or max_tokens.

---

## Weight Format

### Files

- `model-00001-of-00002.safetensors` + `model-00002-of-00002.safetensors`: ~4.5 GB total, BF16
- `model.safetensors.index.json`: weight-to-shard mapping
- `tokenizer.json`: BPE tokenizer (vocab + merges + added_tokens)
- `config.json`: model configuration (vision_config + text_config)

### Tensor Names

**Vision Encoder** (prefix: `model.vision_model.`):
```
embeddings.patch_embedding.weight     [1152, 3, 14, 14] + bias [1152]
embeddings.position_embedding.weight  [729, 1152]
encoder.layers.{i}.layer_norm1.weight [1152] + bias [1152]
encoder.layers.{i}.self_attn.q_proj.weight [1152, 1152] + bias [1152]
encoder.layers.{i}.self_attn.k_proj.weight [1152, 1152] + bias [1152]
encoder.layers.{i}.self_attn.v_proj.weight [1152, 1152] + bias [1152]
encoder.layers.{i}.self_attn.out_proj.weight [1152, 1152] + bias [1152]
encoder.layers.{i}.layer_norm2.weight [1152] + bias [1152]
encoder.layers.{i}.mlp.fc1.weight     [4304, 1152] + bias [4304]
encoder.layers.{i}.mlp.fc2.weight     [1152, 4304] + bias [1152]
post_layernorm.weight                 [1152] + bias [1152]
```

**Connector** (prefix: `model.connector.`):
```
modality_projection.proj.weight       [2048, 10368]
```

**Token Embeddings** (prefix: `model.text_model.`):
```
model.embed_tokens.weight             [49155, 2048]
```

**LM Head** (separate, NOT tied):
```
model.text_model.lm_head.weight       [49155, 2048]
```

**Decoder Layers** (prefix: `model.text_model.model.layers.{i}.`):
```
input_layernorm.weight                [2048]
self_attn.q_proj.weight               [2048, 2048]   (32×64, 2048)
self_attn.k_proj.weight               [2048, 2048]   (32×64, 2048)
self_attn.v_proj.weight               [2048, 2048]   (32×64, 2048)
self_attn.o_proj.weight               [2048, 2048]
post_attention_layernorm.weight       [2048]
mlp.gate_proj.weight                  [8192, 2048]
mlp.up_proj.weight                    [8192, 2048]
mlp.down_proj.weight                  [2048, 8192]
```
Plus `model.text_model.model.norm.weight [2048]` (final norm).

NO biases in any decoder layer.

---

## Kernel Reuse from qwen-asr

SmolVLM reuses the following kernels without modification:

| SmolVLM Operation | Kernel Function |
|---|---|
| SigLIP patch embed (Conv2d k=14 s=14) | `qwen_conv2d` |
| SigLIP LayerNorm (with bias) | `qwen_layer_norm` |
| SigLIP global attention | `qwen_bidirectional_attention` (1 window = full seq) |
| SigLIP GELU MLP | `qwen_linear` + `qwen_gelu` |
| SmolLM2 RMSNorm | `qwen_rms_norm` |
| SmolLM2 QKV/O projections (bf16) | `qwen_linear_nobias_bf16`, `qwen_linear_nobias_bf16_qkv` |
| SmolLM2 NeoX RoPE | `qwen_compute_rope_neox` + `qwen_apply_rope_neox` |
| SmolLM2 causal attention (MHA) | `qwen_causal_attention` (n_kv_heads=n_heads) |
| SmolLM2 SwiGLU | `qwen_swiglu_multiply` + fused gate_up |
| Token generation | `qwen_argmax_matvec_bf16` (against separate lm_head) |

No new kernel functions were needed.

---

## References

- SmolVLM: [HuggingFaceTB/SmolVLM-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)
- SigLIP: [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) (Zhai et al., 2023)
- SmolLM2: [HuggingFaceTB/SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B)
- Idefics3: [HuggingFaceM4/Idefics3-8B-Llama3](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3)
- Kernel infrastructure: [qwen-asr](https://github.com/antirez/qwen-asr) by Salvatore Sanfilippo
