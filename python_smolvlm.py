#!/usr/bin/env python3
"""
python_smolvlm.py - Pure-Python reference implementation for SmolVLM-Instruct inference.

SmolVLM = SigLIP vision encoder + pixel shuffle connector + SmolLM2 decoder.

Usage:
    python3 python_smolvlm.py --model-dir smolvlm-instruct --image test.png --prompt "Describe this image"

Dependencies: numpy, safetensors, PIL (Pillow)
"""

import argparse
import json
import math
import os
import struct
import sys
import time

import numpy as np

# ============================================================================
# Safetensors loading
# ============================================================================

def load_safetensors(path):
    """Load a safetensors file, return dict of name -> numpy array."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size).decode("utf-8")
        header = json.loads(header_json)
        data_start = 8 + header_size
        tensors = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype_str = meta["dtype"]
            shape = meta["data_offsets"]
            begin, end = meta["data_offsets"]
            f.seek(data_start + begin)
            raw = f.read(end - begin)
            if dtype_str == "BF16":
                # Convert bf16 to f32
                u16 = np.frombuffer(raw, dtype=np.uint16)
                u32 = u16.astype(np.uint32) << 16
                arr = u32.view(np.float32).reshape(meta["shape"])
            elif dtype_str == "F32":
                arr = np.frombuffer(raw, dtype=np.float32).reshape(meta["shape"]).copy()
            elif dtype_str == "F16":
                arr = np.frombuffer(raw, dtype=np.float16).reshape(meta["shape"]).astype(np.float32)
            elif dtype_str == "I64":
                arr = np.frombuffer(raw, dtype=np.int64).reshape(meta["shape"]).copy()
            elif dtype_str == "I32":
                arr = np.frombuffer(raw, dtype=np.int32).reshape(meta["shape"]).copy()
            else:
                print(f"Warning: unknown dtype {dtype_str} for {name}, skipping")
                continue
            tensors[name] = arr
        return tensors


def load_model_tensors(model_dir):
    """Load all safetensors shards from model directory."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    single_path = os.path.join(model_dir, "model.safetensors")

    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.loads(f.read())
        shard_files = sorted(set(index["weight_map"].values()))
        tensors = {}
        for shard in shard_files:
            path = os.path.join(model_dir, shard)
            print(f"  Loading {shard}...")
            tensors.update(load_safetensors(path))
        return tensors
    elif os.path.exists(single_path):
        print(f"  Loading model.safetensors...")
        return load_safetensors(single_path)
    else:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")


# ============================================================================
# Tokenizer (GPT-2 BPE from tokenizer.json)
# ============================================================================

class Tokenizer:
    def __init__(self, model_dir):
        tok_path = os.path.join(model_dir, "tokenizer.json")
        with open(tok_path) as f:
            tok_data = json.loads(f.read())

        model = tok_data["model"]
        self.vocab = model["vocab"]  # str -> int
        self.merges = model["merges"]  # list of "a b" strings
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        # Build merge rank map
        self.merge_ranks = {}
        for i, merge in enumerate(self.merges):
            self.merge_ranks[merge] = i

        # Added tokens
        self.added_tokens = {}
        for tok in tok_data.get("added_tokens", []):
            self.added_tokens[tok["content"]] = tok["id"]
            self.id_to_token[tok["id"]] = tok["content"]

        # GPT-2 byte-to-unicode mapping
        self._init_byte_mapping()

    def _init_byte_mapping(self):
        """GPT-2 bytes_to_unicode mapping."""
        bs = list(range(ord('!'), ord('~') + 1)) + \
             list(range(ord('\xa1'), ord('\xac') + 1)) + \
             list(range(ord('\xae'), ord('\xff') + 1))
        cs = list(bs)
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        self.byte_encoder = {b: chr(c) for b, c in zip(bs, cs)}
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

    def _bytes_to_unicode(self, token_bytes):
        return ''.join(self.byte_encoder[b] for b in token_bytes)

    def _unicode_to_bytes(self, token_str):
        return bytes([self.byte_decoder[c] for c in token_str])

    def encode(self, text):
        """Encode text to token IDs using BPE."""
        # Convert bytes to GPT-2 unicode representation
        encoded = self._bytes_to_unicode(text.encode("utf-8"))

        # Split into individual characters
        symbols = list(encoded)
        if len(symbols) <= 1:
            token = encoded
            if token in self.vocab:
                return [self.vocab[token]]
            return []

        # BPE merging
        while len(symbols) > 1:
            best_rank = float('inf')
            best_i = -1
            for i in range(len(symbols) - 1):
                pair = symbols[i] + " " + symbols[i + 1]
                rank = self.merge_ranks.get(pair, float('inf'))
                if rank < best_rank:
                    best_rank = rank
                    best_i = i
            if best_i < 0 or best_rank == float('inf'):
                break
            merged = symbols[best_i] + symbols[best_i + 1]
            symbols = symbols[:best_i] + [merged] + symbols[best_i + 2:]

        ids = []
        for sym in symbols:
            if sym in self.vocab:
                ids.append(self.vocab[sym])
            else:
                # Byte-level fallback (should not happen with proper BPE)
                for ch in sym:
                    if ch in self.vocab:
                        ids.append(self.vocab[ch])
        return ids

    def decode(self, ids):
        """Decode token IDs to text."""
        parts = []
        for tid in ids:
            token_str = self.id_to_token.get(tid, "")
            parts.append(token_str)
        combined = "".join(parts)
        try:
            text = self._unicode_to_bytes(combined).decode("utf-8", errors="replace")
        except Exception:
            text = combined
        return text

    def decode_token(self, tid):
        """Decode a single token ID to text."""
        token_str = self.id_to_token.get(tid, "")
        try:
            text = self._unicode_to_bytes(token_str).decode("utf-8", errors="replace")
        except Exception:
            text = token_str
        return text


# ============================================================================
# Image loading and preprocessing
# ============================================================================

def load_image(path):
    """Load image as numpy array [H, W, 3] uint8."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return np.array(img)


def resize_image(img, target_size):
    """Bilinear resize to target_size x target_size."""
    h, w, c = img.shape
    out = np.zeros((target_size, target_size, c), dtype=np.float32)
    for ch in range(c):
        for y in range(target_size):
            for x in range(target_size):
                src_y = y * (h - 1) / max(target_size - 1, 1)
                src_x = x * (w - 1) / max(target_size - 1, 1)
                y0 = int(src_y)
                x0 = int(src_x)
                y1 = min(y0 + 1, h - 1)
                x1 = min(x0 + 1, w - 1)
                fy = src_y - y0
                fx = src_x - x0
                val = (img[y0, x0, ch] * (1 - fy) * (1 - fx) +
                       img[y1, x0, ch] * fy * (1 - fx) +
                       img[y0, x1, ch] * (1 - fy) * fx +
                       img[y1, x1, ch] * fy * fx)
                out[y, x, ch] = val
    return out


def preprocess_image(img, image_size=384):
    """Resize and normalize image for SigLIP.

    SigLIP normalization: pixel/255 * rescale_factor, then (x - mean) / std
    For SigLIP: rescale_factor=1/255, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]
    Equivalent to: x/255 * 2 - 1, mapping [0,255] to [-1,1]
    """
    if img.dtype == np.uint8:
        img = img.astype(np.float32)

    # Resize
    resized = resize_image(img, image_size)

    # Normalize to [-1, 1]
    resized = resized / 255.0 * 2.0 - 1.0

    # Convert to channel-first: [3, H, W]
    return resized.transpose(2, 0, 1)


# ============================================================================
# Math primitives
# ============================================================================

def gelu(x):
    """GELU activation (tanh approximation as used by SigLIP)."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def silu(x):
    """SiLU (Swish) activation."""
    return x / (1.0 + np.exp(-x))


def layer_norm(x, weight, bias, eps=1e-6):
    """Layer normalization."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def rms_norm(x, weight, eps=1e-5):
    """RMS normalization."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x / rms * weight


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = x.max(axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=axis, keepdims=True)


# ============================================================================
# SigLIP Vision Encoder
# ============================================================================

class SigLIPEncoder:
    def __init__(self, tensors, config):
        vc = config["vision_config"]
        self.hidden_size = vc["hidden_size"]       # 1152
        self.num_heads = vc["num_attention_heads"]  # 16
        self.head_dim = self.hidden_size // self.num_heads  # 72
        self.num_layers = vc["num_hidden_layers"]   # 27
        self.intermediate_size = vc["intermediate_size"]  # 4304
        self.image_size = vc["image_size"]          # 384
        self.patch_size = vc["patch_size"]          # 14
        self.layer_norm_eps = vc.get("layer_norm_eps", 1e-6)

        prefix = "model.vision_model"

        # Patch embedding (Conv2d: [hidden, 3, patch, patch])
        self.patch_weight = tensors[f"{prefix}.embeddings.patch_embedding.weight"]
        self.patch_bias = tensors[f"{prefix}.embeddings.patch_embedding.bias"]

        # Position embeddings (learnable)
        self.position_embedding = tensors[f"{prefix}.embeddings.position_embedding.weight"]

        # Transformer layers
        self.layers = []
        for i in range(self.num_layers):
            lp = f"{prefix}.encoder.layers.{i}"
            layer = {
                "ln1_w": tensors[f"{lp}.layer_norm1.weight"],
                "ln1_b": tensors[f"{lp}.layer_norm1.bias"],
                "q_w": tensors[f"{lp}.self_attn.q_proj.weight"],
                "q_b": tensors[f"{lp}.self_attn.q_proj.bias"],
                "k_w": tensors[f"{lp}.self_attn.k_proj.weight"],
                "k_b": tensors[f"{lp}.self_attn.k_proj.bias"],
                "v_w": tensors[f"{lp}.self_attn.v_proj.weight"],
                "v_b": tensors[f"{lp}.self_attn.v_proj.bias"],
                "o_w": tensors[f"{lp}.self_attn.out_proj.weight"],
                "o_b": tensors[f"{lp}.self_attn.out_proj.bias"],
                "ln2_w": tensors[f"{lp}.layer_norm2.weight"],
                "ln2_b": tensors[f"{lp}.layer_norm2.bias"],
                "fc1_w": tensors[f"{lp}.mlp.fc1.weight"],
                "fc1_b": tensors[f"{lp}.mlp.fc1.bias"],
                "fc2_w": tensors[f"{lp}.mlp.fc2.weight"],
                "fc2_b": tensors[f"{lp}.mlp.fc2.bias"],
            }
            self.layers.append(layer)

        # Post-layernorm
        self.post_ln_w = tensors[f"{prefix}.post_layernorm.weight"]
        self.post_ln_b = tensors[f"{prefix}.post_layernorm.bias"]

    def patch_embed(self, image):
        """Conv2d patch embedding: [3, H, W] -> [num_patches, hidden]"""
        C, H, W = image.shape
        pH = H // self.patch_size  # 27
        pW = W // self.patch_size  # 27

        # Extract patches and project via conv kernel
        # Conv2d with kernel=patch_size, stride=patch_size is equivalent to
        # extracting non-overlapping patches and multiplying by reshaped weight
        # weight shape: [hidden, 3, patch, patch] -> [hidden, 3*patch*patch]
        weight = self.patch_weight.reshape(self.hidden_size, -1)  # [1152, 588]
        patches = np.zeros((pH * pW, C * self.patch_size * self.patch_size), dtype=np.float32)

        for i in range(pH):
            for j in range(pW):
                patch = image[:, i * self.patch_size:(i + 1) * self.patch_size,
                              j * self.patch_size:(j + 1) * self.patch_size]
                patches[i * pW + j] = patch.reshape(-1)

        # [num_patches, 588] @ [588, 1152] + bias
        out = patches @ weight.T + self.patch_bias
        return out  # [num_patches, hidden]

    def attention(self, x, layer):
        """Multi-head self-attention (bidirectional)."""
        seq_len, dim = x.shape
        n_heads = self.num_heads
        head_dim = self.head_dim

        # QKV projections
        q = x @ layer["q_w"].T + layer["q_b"]  # [seq, dim]
        k = x @ layer["k_w"].T + layer["k_b"]
        v = x @ layer["v_w"].T + layer["v_b"]

        # Reshape to [n_heads, seq, head_dim]
        q = q.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
        k = k.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
        v = v.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)

        # Attention scores
        scale = 1.0 / math.sqrt(head_dim)
        scores = (q @ k.transpose(0, 2, 1)) * scale  # [n_heads, seq, seq]
        attn = softmax(scores, axis=-1)
        out = attn @ v  # [n_heads, seq, head_dim]

        # Reshape back
        out = out.transpose(1, 0, 2).reshape(seq_len, dim)

        # Output projection
        out = out @ layer["o_w"].T + layer["o_b"]
        return out

    def forward(self, image):
        """Full encoder forward: image [3, H, W] -> [num_patches, hidden]"""
        print(f"  Vision encoder: {self.num_layers} layers, hidden={self.hidden_size}")

        # Patch embedding
        x = self.patch_embed(image)
        num_patches = x.shape[0]
        print(f"  Patches: {num_patches} ({int(math.sqrt(num_patches))}x{int(math.sqrt(num_patches))})")

        # Add position embeddings
        x = x + self.position_embedding[:num_patches]

        # Transformer layers
        for i, layer in enumerate(self.layers):
            # Pre-LN attention
            x_norm = layer_norm(x, layer["ln1_w"], layer["ln1_b"], self.layer_norm_eps)
            attn_out = self.attention(x_norm, layer)
            x = x + attn_out

            # Pre-LN FFN
            x_norm = layer_norm(x, layer["ln2_w"], layer["ln2_b"], self.layer_norm_eps)
            ffn_out = gelu(x_norm @ layer["fc1_w"].T + layer["fc1_b"])
            ffn_out = ffn_out @ layer["fc2_w"].T + layer["fc2_b"]
            x = x + ffn_out

            if (i + 1) % 9 == 0 or i == self.num_layers - 1:
                print(f"    Layer {i + 1}/{self.num_layers} done")

        # Post-layernorm
        x = layer_norm(x, self.post_ln_w, self.post_ln_b, self.layer_norm_eps)

        return x  # [num_patches, hidden]


# ============================================================================
# Pixel Shuffle Connector
# ============================================================================

class Connector:
    def __init__(self, tensors, config):
        self.scale_factor = config.get("scale_factor", 3)
        prefix = "model.connector"
        self.proj_weight = tensors[f"{prefix}.modality_projection.proj.weight"]

    def pixel_shuffle(self, x, scale_factor):
        """Pixel shuffle: [seq, dim] -> [seq/scale^2, dim*scale^2]"""
        seq_len, embed_dim = x.shape
        h = w = int(math.sqrt(seq_len))
        assert h * w == seq_len, f"Non-square patch grid: {seq_len}"

        x = x.reshape(h, w, embed_dim)
        # Group spatial blocks of scale_factor x scale_factor
        x = x.reshape(h, w // scale_factor, embed_dim * scale_factor)
        x = x.transpose(1, 0, 2)  # [w//sf, h, embed_dim*sf]
        x = x.reshape(w // scale_factor, h // scale_factor, embed_dim * scale_factor * scale_factor)
        x = x.transpose(1, 0, 2)  # [h//sf, w//sf, embed_dim*sf^2]
        x = x.reshape(-1, embed_dim * scale_factor * scale_factor)
        return x

    def forward(self, vision_features):
        """Pixel shuffle + linear projection."""
        print(f"  Connector: pixel_shuffle(scale={self.scale_factor})")
        x = self.pixel_shuffle(vision_features, self.scale_factor)
        print(f"    After shuffle: {x.shape}")
        x = x @ self.proj_weight.T  # Linear, no bias
        print(f"    After projection: {x.shape}")
        return x


# ============================================================================
# SmolLM2 Decoder (Llama-style)
# ============================================================================

class SmolLM2Decoder:
    def __init__(self, tensors, config):
        tc = config["text_config"]
        self.hidden_size = tc["hidden_size"]        # 2048
        self.num_heads = tc["num_attention_heads"]   # 32
        self.num_kv_heads = tc["num_key_value_heads"]  # 32
        self.head_dim = tc.get("head_dim", self.hidden_size // self.num_heads)  # 64
        self.num_layers = tc["num_hidden_layers"]    # 24
        self.intermediate_size = tc["intermediate_size"]  # 8192
        self.rope_theta = tc.get("rope_theta", 273768.0)
        self.rms_norm_eps = tc.get("rms_norm_eps", 1e-5)
        self.vocab_size = tc["vocab_size"]           # 49155

        prefix = "model.text_model.model"

        # Token embeddings
        self.embed_tokens = tensors[f"{prefix}.embed_tokens.weight"]

        # Transformer layers
        self.layers = []
        for i in range(self.num_layers):
            lp = f"{prefix}.layers.{i}"
            layer = {
                "input_norm": tensors[f"{lp}.input_layernorm.weight"],
                "q_w": tensors[f"{lp}.self_attn.q_proj.weight"],
                "k_w": tensors[f"{lp}.self_attn.k_proj.weight"],
                "v_w": tensors[f"{lp}.self_attn.v_proj.weight"],
                "o_w": tensors[f"{lp}.self_attn.o_proj.weight"],
                "post_attn_norm": tensors[f"{lp}.post_attention_layernorm.weight"],
                "gate_w": tensors[f"{lp}.mlp.gate_proj.weight"],
                "up_w": tensors[f"{lp}.mlp.up_proj.weight"],
                "down_w": tensors[f"{lp}.mlp.down_proj.weight"],
            }
            self.layers.append(layer)

        # Final norm
        self.norm = tensors[f"{prefix}.norm.weight"]

        # LM head (separate, not tied)
        self.lm_head = tensors["model.text_model.lm_head.weight"]

        # KV cache
        self.kv_cache_k = None
        self.kv_cache_v = None
        self.kv_len = 0

    def _init_kv_cache(self, max_seq):
        kv_dim = self.num_kv_heads * self.head_dim
        self.kv_cache_k = np.zeros((self.num_layers, max_seq, kv_dim), dtype=np.float32)
        self.kv_cache_v = np.zeros((self.num_layers, max_seq, kv_dim), dtype=np.float32)
        self.kv_len = 0

    def _rope(self, x, positions, n_heads):
        """Apply RoPE (NeoX style: split halves) to x [seq, n_heads*head_dim]."""
        seq_len = x.shape[0]
        x = x.reshape(seq_len, n_heads, self.head_dim)
        half = self.head_dim // 2

        inv_freq = 1.0 / (self.rope_theta ** (np.arange(0, half, dtype=np.float32) * 2.0 / self.head_dim))
        pos = np.array(positions, dtype=np.float32)

        # [seq, half]
        angles = np.outer(pos, inv_freq)

        cos_vals = np.cos(angles)  # [seq, half]
        sin_vals = np.sin(angles)  # [seq, half]

        x_first = x[:, :, :half]   # [seq, n_heads, half]
        x_second = x[:, :, half:]  # [seq, n_heads, half]

        cos_expanded = cos_vals[:, None, :]  # [seq, 1, half]
        sin_expanded = sin_vals[:, None, :]

        r_first = x_first * cos_expanded - x_second * sin_expanded
        r_second = x_first * sin_expanded + x_second * cos_expanded

        result = np.concatenate([r_first, r_second], axis=-1)
        return result.reshape(seq_len, n_heads * self.head_dim)

    def prefill(self, embeddings):
        """Prefill with sequence of embeddings [seq, hidden]."""
        seq_len = embeddings.shape[0]
        if self.kv_cache_k is None:
            self._init_kv_cache(seq_len + 512)

        x = embeddings.copy()
        start_pos = self.kv_len
        positions = list(range(start_pos, start_pos + seq_len))

        n_heads = self.num_heads
        n_kv_heads = self.num_kv_heads
        head_dim = self.head_dim
        q_dim = n_heads * head_dim
        kv_dim = n_kv_heads * head_dim
        scale = 1.0 / math.sqrt(head_dim)
        gqa_ratio = n_heads // n_kv_heads

        for layer_idx, layer in enumerate(self.layers):
            # Input RMSNorm
            x_norm = rms_norm(x, layer["input_norm"], self.rms_norm_eps)

            # QKV
            q = x_norm @ layer["q_w"].T  # [seq, q_dim]
            k = x_norm @ layer["k_w"].T  # [seq, kv_dim]
            v = x_norm @ layer["v_w"].T

            # RoPE
            q = self._rope(q, positions, n_heads)
            k = self._rope(k, positions, n_kv_heads)

            # Store in KV cache
            self.kv_cache_k[layer_idx, start_pos:start_pos + seq_len] = k
            self.kv_cache_v[layer_idx, start_pos:start_pos + seq_len] = v

            # Attention (causal)
            total_seq = start_pos + seq_len
            full_k = self.kv_cache_k[layer_idx, :total_seq]  # [total, kv_dim]
            full_v = self.kv_cache_v[layer_idx, :total_seq]

            # Reshape for attention
            q_heads = q.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)  # [n_heads, seq, hd]
            k_heads = full_k.reshape(total_seq, n_kv_heads, head_dim).transpose(1, 0, 2)
            v_heads = full_v.reshape(total_seq, n_kv_heads, head_dim).transpose(1, 0, 2)

            # GQA expansion (for MHA, gqa_ratio=1, no-op)
            if gqa_ratio > 1:
                k_heads = np.repeat(k_heads, gqa_ratio, axis=0)
                v_heads = np.repeat(v_heads, gqa_ratio, axis=0)

            scores = (q_heads @ k_heads.transpose(0, 2, 1)) * scale  # [n_heads, seq, total]

            # Causal mask
            mask = np.full((seq_len, total_seq), -1e9, dtype=np.float32)
            for i in range(seq_len):
                mask[i, :start_pos + i + 1] = 0.0
            scores = scores + mask[None, :, :]

            attn = softmax(scores, axis=-1)
            attn_out = attn @ v_heads  # [n_heads, seq, hd]
            attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, q_dim)

            # Output projection + residual
            proj = attn_out @ layer["o_w"].T
            x = x + proj

            # Post-attention RMSNorm + SwiGLU MLP
            x_norm = rms_norm(x, layer["post_attn_norm"], self.rms_norm_eps)
            gate = silu(x_norm @ layer["gate_w"].T)
            up = x_norm @ layer["up_w"].T
            ffn_out = (gate * up) @ layer["down_w"].T
            x = x + ffn_out

            if (layer_idx + 1) % 8 == 0 or layer_idx == self.num_layers - 1:
                print(f"    Prefill layer {layer_idx + 1}/{self.num_layers}")

        self.kv_len = start_pos + seq_len

        # Get logits from last position
        x_last = x[-1:]
        x_last = rms_norm(x_last, self.norm, self.rms_norm_eps)
        logits = x_last @ self.lm_head.T
        return int(np.argmax(logits[0]))

    def forward_token(self, token_embed):
        """Single token forward with KV cache. Returns next token ID."""
        x = token_embed.reshape(1, -1)
        pos = self.kv_len

        n_heads = self.num_heads
        n_kv_heads = self.num_kv_heads
        head_dim = self.head_dim
        q_dim = n_heads * head_dim
        kv_dim = n_kv_heads * head_dim
        scale = 1.0 / math.sqrt(head_dim)
        gqa_ratio = n_heads // n_kv_heads

        for layer_idx, layer in enumerate(self.layers):
            x_norm = rms_norm(x, layer["input_norm"], self.rms_norm_eps)

            q = x_norm @ layer["q_w"].T
            k = x_norm @ layer["k_w"].T
            v = x_norm @ layer["v_w"].T

            q = self._rope(q, [pos], n_heads)
            k = self._rope(k, [pos], n_kv_heads)

            self.kv_cache_k[layer_idx, pos] = k[0]
            self.kv_cache_v[layer_idx, pos] = v[0]

            total_seq = pos + 1
            full_k = self.kv_cache_k[layer_idx, :total_seq]
            full_v = self.kv_cache_v[layer_idx, :total_seq]

            q_heads = q.reshape(1, n_heads, head_dim).transpose(1, 0, 2)
            k_heads = full_k.reshape(total_seq, n_kv_heads, head_dim).transpose(1, 0, 2)
            v_heads = full_v.reshape(total_seq, n_kv_heads, head_dim).transpose(1, 0, 2)

            if gqa_ratio > 1:
                k_heads = np.repeat(k_heads, gqa_ratio, axis=0)
                v_heads = np.repeat(v_heads, gqa_ratio, axis=0)

            scores = (q_heads @ k_heads.transpose(0, 2, 1)) * scale
            attn = softmax(scores, axis=-1)
            attn_out = attn @ v_heads
            attn_out = attn_out.transpose(1, 0, 2).reshape(1, q_dim)

            proj = attn_out @ layer["o_w"].T
            x = x + proj

            x_norm = rms_norm(x, layer["post_attn_norm"], self.rms_norm_eps)
            gate = silu(x_norm @ layer["gate_w"].T)
            up = x_norm @ layer["up_w"].T
            ffn_out = (gate * up) @ layer["down_w"].T
            x = x + ffn_out

        self.kv_len = pos + 1

        x = rms_norm(x, self.norm, self.rms_norm_eps)
        logits = x @ self.lm_head.T
        return int(np.argmax(logits[0]))

    def embed_token(self, token_id):
        """Get embedding for a single token."""
        return self.embed_tokens[token_id]

    def embed_tokens_batch(self, token_ids):
        """Get embeddings for a batch of token IDs."""
        return self.embed_tokens[token_ids]  # [seq, hidden]


# ============================================================================
# SmolVLM Model
# ============================================================================

# Special token IDs
TOKEN_IM_START = 1        # <|im_start|>
TOKEN_IM_END = 2          # <|im_end|>
TOKEN_FAKE_IMAGE = 49152  # <fake_token_around_image>
TOKEN_IMAGE = 49153       # <image>
TOKEN_EOS = 49154         # <end_of_utterance>


class SmolVLM:
    def __init__(self, model_dir):
        print(f"Loading SmolVLM from {model_dir}/")

        # Load config
        with open(os.path.join(model_dir, "config.json")) as f:
            self.config = json.loads(f.read())

        self.image_seq_len = self.config.get("image_seq_len", 81)
        self.scale_factor = self.config.get("scale_factor", 3)

        # Load weights
        t0 = time.time()
        tensors = load_model_tensors(model_dir)
        print(f"  Loaded {len(tensors)} tensors in {time.time() - t0:.1f}s")

        # Initialize components
        self.vision = SigLIPEncoder(tensors, self.config)
        self.connector = Connector(tensors, self.config)
        self.decoder = SmolLM2Decoder(tensors, self.config)
        self.tokenizer = Tokenizer(model_dir)

        print(f"  Vision: {self.vision.num_layers} layers, hidden={self.vision.hidden_size}")
        print(f"  Decoder: {self.decoder.num_layers} layers, hidden={self.decoder.hidden_size}")
        print(f"  Vocab: {self.decoder.vocab_size}")

    def build_prompt_tokens(self, prompt_text):
        """Build the prompt token sequence with image placeholder."""
        # Format: <|im_start|>User:<fake_image><image>...<image><fake_image>prompt<eos>\nAssistant:
        tokens = [TOKEN_IM_START]

        # Encode "User:"
        user_tokens = self.tokenizer.encode("User:")
        tokens.extend(user_tokens)

        # Image tokens
        tokens.append(TOKEN_FAKE_IMAGE)
        for _ in range(self.image_seq_len):
            tokens.append(TOKEN_IMAGE)
        tokens.append(TOKEN_FAKE_IMAGE)

        # Prompt text
        prompt_tokens = self.tokenizer.encode(prompt_text)
        tokens.extend(prompt_tokens)

        # End of utterance + newline
        tokens.append(TOKEN_EOS)
        newline_tokens = self.tokenizer.encode("\n")
        tokens.extend(newline_tokens)

        # Assistant prompt
        assistant_tokens = self.tokenizer.encode("Assistant:")
        tokens.extend(assistant_tokens)

        return tokens

    def generate(self, image_path, prompt, max_tokens=256):
        """Generate text from image + prompt."""
        # Load and preprocess image
        print("\n--- Image Processing ---")
        img = load_image(image_path)
        print(f"  Input image: {img.shape}")
        image_tensor = preprocess_image(img, self.vision.image_size)
        print(f"  Preprocessed: {image_tensor.shape}")

        # Vision encoder
        print("\n--- Vision Encoder ---")
        t0 = time.time()
        vision_features = self.vision.forward(image_tensor)
        enc_time = time.time() - t0
        print(f"  Vision output: {vision_features.shape} ({enc_time:.1f}s)")

        # Connector
        print("\n--- Connector ---")
        image_embeds = self.connector.forward(vision_features)
        print(f"  Image embeddings: {image_embeds.shape}")

        # Build prompt
        print("\n--- Token Sequence ---")
        token_ids = self.build_prompt_tokens(prompt)
        print(f"  Prompt tokens: {len(token_ids)}")

        # Build embedding sequence: replace image tokens with vision embeddings
        embeddings = np.zeros((len(token_ids), self.decoder.hidden_size), dtype=np.float32)
        img_idx = 0
        for i, tid in enumerate(token_ids):
            if tid == TOKEN_IMAGE:
                if img_idx < image_embeds.shape[0]:
                    embeddings[i] = image_embeds[img_idx]
                    img_idx += 1
            else:
                embeddings[i] = self.decoder.embed_token(tid)

        # Prefill all-but-last, get first token from last position
        print(f"\n--- Decoding ---")
        print(f"  Prefilling {len(token_ids)} tokens...")
        t_dec_start = time.time()

        # Prefill all tokens except the last one, then the last position predicts first generated token
        first_token = self.decoder.prefill(embeddings)
        print(f"  First token: {first_token} = '{self.tokenizer.decode_token(first_token)}'")

        # Autoregressive generation
        generated = [first_token]
        sys.stdout.write(self.tokenizer.decode_token(first_token))
        sys.stdout.flush()

        for step in range(max_tokens - 1):
            token_embed = self.decoder.embed_token(generated[-1])
            next_token = self.decoder.forward_token(token_embed)

            if next_token == TOKEN_EOS:
                break

            generated.append(next_token)
            piece = self.tokenizer.decode_token(next_token)
            sys.stdout.write(piece)
            sys.stdout.flush()

        dec_time = time.time() - t_dec_start
        text = self.tokenizer.decode(generated)
        print(f"\n\n--- Summary ---")
        print(f"  Encoding: {enc_time:.1f}s")
        print(f"  Decoding: {dec_time:.1f}s ({len(generated)} tokens, "
              f"{len(generated)/dec_time:.2f} tok/s)")
        print(f"  Total: {enc_time + dec_time:.1f}s")

        return text


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SmolVLM-Instruct reference inference")
    parser.add_argument("--model-dir", required=True, help="Model directory")
    parser.add_argument("--image", required=True, help="Input image (PNG/JPG)")
    parser.add_argument("--prompt", default="Describe this image in detail.",
                        help="Text prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--dump-tensors", action="store_true",
                        help="Print all tensor names and shapes, then exit")
    args = parser.parse_args()

    if args.dump_tensors:
        tensors = load_model_tensors(args.model_dir)
        for name in sorted(tensors.keys()):
            t = tensors[name]
            print(f"  {name}: {t.shape} {t.dtype}")
        return

    model = SmolVLM(args.model_dir)
    text = model.generate(args.image, args.prompt, args.max_tokens)
    print(f"\nFinal output: {text}")


if __name__ == "__main__":
    main()
