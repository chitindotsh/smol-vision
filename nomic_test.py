#!/usr/bin/env python3
"""
nomic_embed test suite — kernel correctness and FFI boundary tests.

Tests the Q4 dequantization and mean pooling NEON kernels against
Python reference implementations via ctypes.  Builds a shared library
from the C sources, then exercises each kernel with known inputs.

Usage:
  ./nomic_test.py                    # Run all kernel tests
  ./nomic_test.py --rebuild          # Force rebuild of shared library
  ./nomic_test.py --verbose          # Print per-element comparisons

Requires: numpy
"""

from __future__ import annotations

import argparse
import ctypes
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ---- ANSI colors ----

_USE_COLOR = (
    hasattr(sys.stdout, "isatty")
    and sys.stdout.isatty()
    and os.environ.get("NO_COLOR") is None
)

def _sgr(code: str) -> str:
    return f"\033[{code}m" if _USE_COLOR else ""

C_RESET   = _sgr("0")
C_BOLD    = _sgr("1")
C_DIM     = _sgr("2")
C_RED     = _sgr("31")
C_GREEN   = _sgr("32")
C_YELLOW  = _sgr("33")
C_CYAN    = _sgr("36")
C_BRED    = _sgr("1;31")
C_BGREEN  = _sgr("1;32")
C_BYELLOW = _sgr("1;33")
C_BCYAN   = _sgr("1;36")
C_BWHITE  = _sgr("1;37")

# ---- Build shared library ----

SCRIPT_DIR = Path(__file__).resolve().parent

# Source files needed for the kernel-only shared library.
# We compile nomic_kernels_neon.c plus nomic_embed.c and all qwen
# dependencies so the linker resolves every symbol.
KERNEL_LIB_SRCS = [
    "nomic_kernels_neon.c",
    "nomic_embed.c",
    "qwen_asr.c",
    "qwen_asr_kernels.c",
    "qwen_asr_kernels_generic.c",
    "qwen_asr_kernels_neon.c",
    "qwen_asr_kernels_avx.c",
    "qwen_asr_audio.c",
    "qwen_asr_encoder.c",
    "qwen_asr_decoder.c",
    "qwen_asr_safetensors.c",
    "qwen_asr_tokenizer.c",
    "qwen25_omni.c",
    "qwen25_omni_encoder.c",
    "qwen25_omni_decoder.c",
]

LIB_NAME = "libnomic_test.dylib" if sys.platform == "darwin" else "libnomic_test.so"


def build_shared_lib(force: bool = False) -> Path:
    """Compile the nomic kernel sources into a shared library for ctypes."""
    lib_path = SCRIPT_DIR / LIB_NAME
    srcs = [SCRIPT_DIR / s for s in KERNEL_LIB_SRCS]

    # Skip rebuild if library is newer than all sources
    if not force and lib_path.exists():
        lib_mtime = lib_path.stat().st_mtime
        if all(s.stat().st_mtime < lib_mtime for s in srcs if s.exists()):
            return lib_path

    print(f"{C_BCYAN}[BUILD]{C_RESET} Compiling {LIB_NAME}...", flush=True)

    cc = os.environ.get("CC", "gcc")
    cflags = ["-shared", "-fPIC", "-O2", "-march=native", "-ffast-math"]
    if sys.platform == "darwin":
        cflags += ["-dynamiclib", "-DUSE_BLAS", "-DACCELERATE_NEW_LAPACK"]
        ldflags = ["-lm", "-lpthread", "-framework", "Accelerate"]
    else:
        cflags += ["-DUSE_BLAS", "-DUSE_OPENBLAS", "-I/usr/include/openblas"]
        ldflags = ["-lm", "-lpthread", "-lopenblas"]

    cmd = [cc] + cflags + [str(s) for s in srcs] + ["-o", str(lib_path)] + ldflags
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SCRIPT_DIR))
    if result.returncode != 0:
        print(f"{C_BRED}[BUILD FAILED]{C_RESET}")
        print(result.stderr)
        sys.exit(1)

    print(f"{C_BGREEN}[BUILD OK]{C_RESET} {lib_path.name}")
    return lib_path


def load_lib(lib_path: Path) -> ctypes.CDLL:
    """Load the shared library and set up function prototypes."""
    lib = ctypes.CDLL(str(lib_path))

    # void nomic_dequantize_q4_neon(float *out, const uint8_t *packed,
    #                                const uint16_t *scales_f16,
    #                                int n, int block_size);
    lib.nomic_dequantize_q4_neon.restype = None
    lib.nomic_dequantize_q4_neon.argtypes = [
        ctypes.POINTER(ctypes.c_float),     # out
        ctypes.POINTER(ctypes.c_uint8),     # packed
        ctypes.POINTER(ctypes.c_uint16),    # scales_f16
        ctypes.c_int,                       # n
        ctypes.c_int,                       # block_size
    ]

    # void nomic_mean_pool_neon(float *out, const float *hidden_states,
    #                           const int *seq_starts, const int *seq_lens,
    #                           int num_seqs, int hidden);
    lib.nomic_mean_pool_neon.restype = None
    lib.nomic_mean_pool_neon.argtypes = [
        ctypes.POINTER(ctypes.c_float),     # out
        ctypes.POINTER(ctypes.c_float),     # hidden_states
        ctypes.POINTER(ctypes.c_int),       # seq_starts
        ctypes.POINTER(ctypes.c_int),       # seq_lens
        ctypes.c_int,                       # num_seqs
        ctypes.c_int,                       # hidden
    ]

    return lib


# ---- Python reference implementations ----

def ref_dequantize_q4(packed: np.ndarray, scales_f16: np.ndarray,
                       n: int, block_size: int) -> np.ndarray:
    """Reference Q4 dequantization in pure Python/numpy."""
    out = np.zeros(n, dtype=np.float32)
    num_blocks = n // block_size
    bytes_per_block = block_size // 2
    scales = scales_f16.astype(np.float32)

    for b in range(num_blocks):
        scale = scales[b]
        src = packed[b * bytes_per_block : (b + 1) * bytes_per_block]
        dst_off = b * block_size
        for i, byte in enumerate(src):
            lo = int(byte) & 0x0F
            hi = int(byte) >> 4
            out[dst_off + i * 2]     = (lo - 8) * scale
            out[dst_off + i * 2 + 1] = (hi - 8) * scale

    return out


def ref_mean_pool(hidden_states: np.ndarray, seq_starts: List[int],
                   seq_lens: List[int], hidden: int) -> np.ndarray:
    """Reference mean pooling in numpy."""
    num_seqs = len(seq_starts)
    out = np.zeros((num_seqs, hidden), dtype=np.float32)
    for s in range(num_seqs):
        start = seq_starts[s]
        length = seq_lens[s]
        if length > 0:
            chunk = hidden_states[start : start + length, :]
            out[s] = chunk.mean(axis=0)
    return out


# ---- ctypes helpers ----

def np_to_ctypes_float(arr: np.ndarray):
    """Get a ctypes float pointer from a contiguous float32 numpy array."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def np_to_ctypes_uint8(arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


def np_to_ctypes_uint16(arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.uint16)
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))


def np_to_ctypes_int(arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.int32)
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


def alloc_float(n: int) -> Tuple[np.ndarray, ctypes.POINTER(ctypes.c_float)]:
    """Allocate a float32 array and return both numpy view and ctypes pointer."""
    arr = np.zeros(n, dtype=np.float32)
    ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    return arr, ptr


# ---- Test runner infrastructure ----

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures: List[str] = []

    def ok(self, name: str, detail: str = ""):
        self.passed += 1
        extra = f" {C_DIM}({detail}){C_RESET}" if detail else ""
        print(f"  {C_BGREEN}[PASS]{C_RESET} {name}{extra}")

    def fail(self, name: str, detail: str = ""):
        self.failed += 1
        self.failures.append(name)
        extra = f" {C_DIM}({detail}){C_RESET}" if detail else ""
        print(f"  {C_BRED}[FAIL]{C_RESET} {name}{extra}")

    def skip(self, name: str, reason: str = ""):
        self.skipped += 1
        extra = f" {C_DIM}({reason}){C_RESET}" if reason else ""
        print(f"  {C_BYELLOW}[SKIP]{C_RESET} {name}{extra}")

    def summary(self) -> int:
        total = self.passed + self.failed + self.skipped
        print()
        if self.failed == 0:
            print(
                f"{C_BGREEN}All {self.passed} tests passed{C_RESET}"
                + (f" ({self.skipped} skipped)" if self.skipped else "")
            )
            return 0
        else:
            print(
                f"{C_BRED}{self.failed} FAILED{C_RESET}, "
                f"{self.passed} passed"
                + (f", {self.skipped} skipped" if self.skipped else "")
            )
            for name in self.failures:
                print(f"  - {name}")
            return 1


def assert_close(name: str, actual: np.ndarray, expected: np.ndarray,
                  results: TestResults, atol: float = 1e-5, rtol: float = 1e-4,
                  verbose: bool = False) -> bool:
    """Compare two arrays, report pass/fail."""
    if actual.shape != expected.shape:
        results.fail(name, f"shape mismatch: {actual.shape} vs {expected.shape}")
        return False

    abs_diff = np.abs(actual - expected)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    close = np.allclose(actual, expected, atol=atol, rtol=rtol)

    if verbose and not close:
        # Show first few mismatches
        mask = ~np.isclose(actual, expected, atol=atol, rtol=rtol)
        indices = np.where(mask.ravel())[0][:5]
        for idx in indices:
            print(
                f"       [{idx}] actual={actual.ravel()[idx]:.6f} "
                f"expected={expected.ravel()[idx]:.6f} "
                f"diff={abs_diff.ravel()[idx]:.2e}"
            )

    if close:
        results.ok(name, f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
        return True
    else:
        results.fail(name, f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
        return False


# ---- Q4 Dequantization Tests ----

def test_dequant_basic(lib, results: TestResults, verbose: bool):
    """Single block of 32 elements with a known scale."""
    block_size = 32
    n = block_size
    bytes_per_block = block_size // 2  # 16 bytes

    # Build packed bytes: byte[i] = (hi << 4) | lo
    # Use a simple pattern: lo nibble = i % 16, hi nibble = (i + 1) % 16
    packed = np.zeros(bytes_per_block, dtype=np.uint8)
    for i in range(bytes_per_block):
        lo = i % 16
        hi = (i + 1) % 16
        packed[i] = (hi << 4) | lo

    # Scale = 0.5 in f16
    scales = np.array([0.5], dtype=np.float16)
    scales_u16 = scales.view(np.uint16)

    # Reference
    expected = ref_dequantize_q4(packed, scales.astype(np.float32), n, block_size)

    # C kernel
    out, out_ptr = alloc_float(n)
    lib.nomic_dequantize_q4_neon(
        out_ptr,
        np_to_ctypes_uint8(packed),
        np_to_ctypes_uint16(scales_u16),
        n,
        block_size,
    )

    assert_close("dequant_basic (32 elems, scale=0.5)", out, expected,
                  results, verbose=verbose)


def test_dequant_multi_block(lib, results: TestResults, verbose: bool):
    """Multiple blocks with different scales."""
    block_size = 32
    num_blocks = 4
    n = num_blocks * block_size
    bytes_per_block = block_size // 2

    # Random packed bytes
    rng = np.random.RandomState(42)
    packed = rng.randint(0, 256, size=num_blocks * bytes_per_block, dtype=np.uint8)

    # Different scale per block: 0.25, 0.5, 1.0, 2.0
    scales = np.array([0.25, 0.5, 1.0, 2.0], dtype=np.float16)
    scales_u16 = scales.view(np.uint16)

    expected = ref_dequantize_q4(packed, scales.astype(np.float32), n, block_size)

    out, out_ptr = alloc_float(n)
    lib.nomic_dequantize_q4_neon(
        out_ptr,
        np_to_ctypes_uint8(packed),
        np_to_ctypes_uint16(scales_u16),
        n,
        block_size,
    )

    assert_close("dequant_multi_block (4 blocks, varied scales)", out, expected,
                  results, verbose=verbose)


def test_dequant_zero_scale(lib, results: TestResults, verbose: bool):
    """Scale = 0 should produce all zeros regardless of packed data."""
    block_size = 32
    n = block_size

    packed = np.full(block_size // 2, 0xFF, dtype=np.uint8)  # all nibbles = 15
    scales = np.array([0.0], dtype=np.float16)
    scales_u16 = scales.view(np.uint16)

    expected = np.zeros(n, dtype=np.float32)

    out, out_ptr = alloc_float(n)
    lib.nomic_dequantize_q4_neon(
        out_ptr,
        np_to_ctypes_uint8(packed),
        np_to_ctypes_uint16(scales_u16),
        n,
        block_size,
    )

    assert_close("dequant_zero_scale (all zeros expected)", out, expected,
                  results, verbose=verbose)


def test_dequant_negative_weights(lib, results: TestResults, verbose: bool):
    """Verify correct centering: nibble 0 -> -8, nibble 7 -> -1, nibble 8 -> 0."""
    block_size = 32
    n = block_size

    # All bytes = 0x80: lo=0 (weight=-8), hi=8 (weight=0)
    packed = np.full(block_size // 2, 0x80, dtype=np.uint8)
    scales = np.array([1.0], dtype=np.float16)
    scales_u16 = scales.view(np.uint16)

    expected = ref_dequantize_q4(packed, scales.astype(np.float32), n, block_size)

    out, out_ptr = alloc_float(n)
    lib.nomic_dequantize_q4_neon(
        out_ptr,
        np_to_ctypes_uint8(packed),
        np_to_ctypes_uint16(scales_u16),
        n,
        block_size,
    )

    # Verify specific values
    ok = assert_close("dequant_negative_weights (centering check)", out, expected,
                       results, verbose=verbose)
    if ok:
        # Double-check the pattern: even indices = -8, odd indices = 0
        if not (np.all(out[0::2] == -8.0) and np.all(out[1::2] == 0.0)):
            results.fail("dequant_negative_weights (value spot check)",
                         f"even={out[0]}, odd={out[1]}")


def test_dequant_large_matrix(lib, results: TestResults, verbose: bool):
    """Realistic weight matrix size: [768, 768] = 589824 elements."""
    block_size = 32
    rows, cols = 768, 768
    n = rows * cols
    num_blocks = n // block_size

    rng = np.random.RandomState(123)
    packed = rng.randint(0, 256, size=n // 2, dtype=np.uint8)
    scales = rng.uniform(-2.0, 2.0, size=num_blocks).astype(np.float16)
    scales_u16 = scales.view(np.uint16)

    expected = ref_dequantize_q4(packed, scales.astype(np.float32), n, block_size)

    out, out_ptr = alloc_float(n)
    t0 = time.monotonic()
    lib.nomic_dequantize_q4_neon(
        out_ptr,
        np_to_ctypes_uint8(packed),
        np_to_ctypes_uint16(scales_u16),
        n,
        block_size,
    )
    elapsed = time.monotonic() - t0

    assert_close(
        f"dequant_large_matrix (768x768={n} elems, {elapsed*1000:.1f}ms)",
        out, expected, results, atol=1e-3, verbose=verbose,
    )


def test_dequant_block_size_64(lib, results: TestResults, verbose: bool):
    """Non-default block size of 64."""
    block_size = 64
    num_blocks = 2
    n = num_blocks * block_size
    bytes_per_block = block_size // 2

    rng = np.random.RandomState(77)
    packed = rng.randint(0, 256, size=num_blocks * bytes_per_block, dtype=np.uint8)
    scales = np.array([0.125, 3.0], dtype=np.float16)
    scales_u16 = scales.view(np.uint16)

    expected = ref_dequantize_q4(packed, scales.astype(np.float32), n, block_size)

    out, out_ptr = alloc_float(n)
    lib.nomic_dequantize_q4_neon(
        out_ptr,
        np_to_ctypes_uint8(packed),
        np_to_ctypes_uint16(scales_u16),
        n,
        block_size,
    )

    assert_close("dequant_block_size_64 (2 blocks)", out, expected,
                  results, verbose=verbose)


# ---- Mean Pooling Tests ----

def test_mean_pool_single(lib, results: TestResults, verbose: bool):
    """Single sequence, verify against numpy mean."""
    hidden = 16
    seq_len = 5

    rng = np.random.RandomState(10)
    hidden_states = rng.randn(seq_len, hidden).astype(np.float32)
    seq_starts = np.array([0], dtype=np.int32)
    seq_lens = np.array([seq_len], dtype=np.int32)

    expected = ref_mean_pool(hidden_states, [0], [seq_len], hidden)

    out, out_ptr = alloc_float(hidden)
    lib.nomic_mean_pool_neon(
        out_ptr,
        np_to_ctypes_float(hidden_states.ravel()),
        np_to_ctypes_int(seq_starts),
        np_to_ctypes_int(seq_lens),
        1,
        hidden,
    )

    assert_close("mean_pool_single (5 tokens, hidden=16)", out, expected.ravel(),
                  results, verbose=verbose)


def test_mean_pool_batch(lib, results: TestResults, verbose: bool):
    """Batch of 3 sequences with different lengths."""
    hidden = 32
    lens = [3, 7, 2]
    total = sum(lens)

    rng = np.random.RandomState(20)
    hidden_states = rng.randn(total, hidden).astype(np.float32)

    starts = []
    offset = 0
    for l in lens:
        starts.append(offset)
        offset += l

    seq_starts = np.array(starts, dtype=np.int32)
    seq_lens = np.array(lens, dtype=np.int32)

    expected = ref_mean_pool(hidden_states, starts, lens, hidden)

    out, out_ptr = alloc_float(len(lens) * hidden)
    lib.nomic_mean_pool_neon(
        out_ptr,
        np_to_ctypes_float(hidden_states.ravel()),
        np_to_ctypes_int(seq_starts),
        np_to_ctypes_int(seq_lens),
        len(lens),
        hidden,
    )

    out_2d = out.reshape(len(lens), hidden)
    assert_close("mean_pool_batch (3 seqs, lens=[3,7,2], hidden=32)",
                  out_2d, expected, results, verbose=verbose)


def test_mean_pool_hidden_768(lib, results: TestResults, verbose: bool):
    """Production hidden size (768), exercises full NEON vectorization."""
    hidden = 768
    lens = [10, 1, 50]
    total = sum(lens)

    rng = np.random.RandomState(30)
    hidden_states = rng.randn(total, hidden).astype(np.float32)

    starts = []
    offset = 0
    for l in lens:
        starts.append(offset)
        offset += l

    seq_starts = np.array(starts, dtype=np.int32)
    seq_lens = np.array(lens, dtype=np.int32)

    expected = ref_mean_pool(hidden_states, starts, lens, hidden)

    out, out_ptr = alloc_float(len(lens) * hidden)
    t0 = time.monotonic()
    lib.nomic_mean_pool_neon(
        out_ptr,
        np_to_ctypes_float(hidden_states.ravel()),
        np_to_ctypes_int(seq_starts),
        np_to_ctypes_int(seq_lens),
        len(lens),
        hidden,
    )
    elapsed = time.monotonic() - t0

    out_2d = out.reshape(len(lens), hidden)
    assert_close(
        f"mean_pool_hidden_768 (3 seqs, {elapsed*1000:.2f}ms)",
        out_2d, expected, results, atol=1e-4, verbose=verbose,
    )


def test_mean_pool_odd_hidden(lib, results: TestResults, verbose: bool):
    """Hidden dim not aligned to 8 — exercises scalar tail in NEON kernel."""
    hidden = 11
    seq_len = 4

    rng = np.random.RandomState(40)
    hidden_states = rng.randn(seq_len, hidden).astype(np.float32)
    seq_starts = np.array([0], dtype=np.int32)
    seq_lens = np.array([seq_len], dtype=np.int32)

    expected = ref_mean_pool(hidden_states, [0], [seq_len], hidden)

    out, out_ptr = alloc_float(hidden)
    lib.nomic_mean_pool_neon(
        out_ptr,
        np_to_ctypes_float(hidden_states.ravel()),
        np_to_ctypes_int(seq_starts),
        np_to_ctypes_int(seq_lens),
        1,
        hidden,
    )

    assert_close("mean_pool_odd_hidden (hidden=11, scalar tail)", out,
                  expected.ravel(), results, verbose=verbose)


def test_mean_pool_single_token(lib, results: TestResults, verbose: bool):
    """Sequence of length 1 — mean should equal the single token."""
    hidden = 64

    rng = np.random.RandomState(50)
    hidden_states = rng.randn(1, hidden).astype(np.float32)
    seq_starts = np.array([0], dtype=np.int32)
    seq_lens = np.array([1], dtype=np.int32)

    expected = hidden_states[0].copy()

    out, out_ptr = alloc_float(hidden)
    lib.nomic_mean_pool_neon(
        out_ptr,
        np_to_ctypes_float(hidden_states.ravel()),
        np_to_ctypes_int(seq_starts),
        np_to_ctypes_int(seq_lens),
        1,
        hidden,
    )

    assert_close("mean_pool_single_token (len=1 -> identity)", out, expected,
                  results, verbose=verbose)


def test_mean_pool_uniform(lib, results: TestResults, verbose: bool):
    """All tokens identical — mean should equal any single token exactly."""
    hidden = 768
    seq_len = 20

    rng = np.random.RandomState(60)
    token = rng.randn(hidden).astype(np.float32)
    hidden_states = np.tile(token, (seq_len, 1))
    seq_starts = np.array([0], dtype=np.int32)
    seq_lens = np.array([seq_len], dtype=np.int32)

    expected = token.copy()

    out, out_ptr = alloc_float(hidden)
    lib.nomic_mean_pool_neon(
        out_ptr,
        np_to_ctypes_float(hidden_states.ravel()),
        np_to_ctypes_int(seq_starts),
        np_to_ctypes_int(seq_lens),
        1,
        hidden,
    )

    assert_close("mean_pool_uniform (all tokens same -> exact mean)", out,
                  expected, results, atol=1e-5, verbose=verbose)


# ---- Q4 Linear round-trip test ----

def test_dequant_symmetry(lib, results: TestResults, verbose: bool):
    """Verify nibble=8 (zero point) produces exactly 0.0 for any scale."""
    block_size = 32
    n = block_size

    # byte 0x88: lo=8 (weight=0), hi=8 (weight=0)
    packed = np.full(block_size // 2, 0x88, dtype=np.uint8)
    scales = np.array([42.0], dtype=np.float16)
    scales_u16 = scales.view(np.uint16)

    expected = np.zeros(n, dtype=np.float32)

    out, out_ptr = alloc_float(n)
    lib.nomic_dequantize_q4_neon(
        out_ptr,
        np_to_ctypes_uint8(packed),
        np_to_ctypes_uint16(scales_u16),
        n,
        block_size,
    )

    assert_close("dequant_symmetry (nibble=8 -> all zeros)", out, expected,
                  results, atol=0.0, verbose=verbose)


def test_dequant_extremes(lib, results: TestResults, verbose: bool):
    """Min/max nibble values: 0 -> -8*scale, 15 -> +7*scale."""
    block_size = 32
    n = block_size

    # All bytes 0x0F: lo=15 (weight=+7), hi=0 (weight=-8)
    packed = np.full(block_size // 2, 0x0F, dtype=np.uint8)
    scale_val = 0.25
    scales = np.array([scale_val], dtype=np.float16)
    scales_u16 = scales.view(np.uint16)

    expected = ref_dequantize_q4(packed, scales.astype(np.float32), n, block_size)

    out, out_ptr = alloc_float(n)
    lib.nomic_dequantize_q4_neon(
        out_ptr,
        np_to_ctypes_uint8(packed),
        np_to_ctypes_uint16(scales_u16),
        n,
        block_size,
    )

    ok = assert_close("dequant_extremes (nibbles 0 and 15)", out, expected,
                       results, verbose=verbose)
    if ok:
        # Spot check: even indices should be +7*0.25=1.75, odd should be -8*0.25=-2.0
        scale_f32 = np.float16(scale_val).astype(np.float32)
        exp_even = 7.0 * scale_f32
        exp_odd = -8.0 * scale_f32
        if not (np.allclose(out[0::2], exp_even) and np.allclose(out[1::2], exp_odd)):
            results.fail("dequant_extremes (spot check)",
                         f"even[0]={out[0]}, odd[0]={out[1]}")


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(
        description="nomic_embed kernel test suite",
    )
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild of shared library")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-element details on failure")
    args = parser.parse_args()

    print(f"{C_BCYAN}{'=' * 60}{C_RESET}")
    print(f"{C_BCYAN}  nomic_embed kernel test suite{C_RESET}")
    print(f"{C_BCYAN}{'=' * 60}{C_RESET}")
    print()

    # Build and load shared library
    lib_path = build_shared_lib(force=args.rebuild)
    lib = load_lib(lib_path)

    results = TestResults()

    # Q4 Dequantization tests
    print(f"{C_BWHITE}Q4 Dequantization Kernel{C_RESET}")
    test_dequant_basic(lib, results, args.verbose)
    test_dequant_multi_block(lib, results, args.verbose)
    test_dequant_zero_scale(lib, results, args.verbose)
    test_dequant_negative_weights(lib, results, args.verbose)
    test_dequant_symmetry(lib, results, args.verbose)
    test_dequant_extremes(lib, results, args.verbose)
    test_dequant_block_size_64(lib, results, args.verbose)
    test_dequant_large_matrix(lib, results, args.verbose)
    print()

    # Mean Pooling tests
    print(f"{C_BWHITE}Mean Pooling Kernel{C_RESET}")
    test_mean_pool_single(lib, results, args.verbose)
    test_mean_pool_batch(lib, results, args.verbose)
    test_mean_pool_hidden_768(lib, results, args.verbose)
    test_mean_pool_odd_hidden(lib, results, args.verbose)
    test_mean_pool_single_token(lib, results, args.verbose)
    test_mean_pool_uniform(lib, results, args.verbose)
    print()

    sys.exit(results.summary())


if __name__ == "__main__":
    main()
