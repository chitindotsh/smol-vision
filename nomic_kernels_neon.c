/*
 * nomic_kernels_neon.c - ARM NEON kernels for nomic-embed-text inference
 *
 * Two kernels required by the embedding engine:
 *   1. Q4 dequantization: packed uint8 nibbles + f16 block scales -> f32
 *   2. Mean pooling: sum hidden states per sequence, divide by length
 *
 * Style: explicit for loops, standard NEON intrinsics, no opaque macros.
 */

#include "nomic_embed.h"
#include <string.h>

#ifdef __ARM_NEON

#include <arm_neon.h>

/* ========================================================================
 * Q4 Dequantization Kernel (NEON)
 *
 * Each packed byte holds two 4-bit weights:
 *   low nibble  = weight at even index
 *   high nibble = weight at odd index
 *
 * Values are unsigned 0-15, centered by subtracting 8 to give signed -8..+7.
 * Each block of block_size elements shares one f16 scale.
 *
 * Processing: 16 packed bytes -> 32 f32 values per inner iteration.
 * ======================================================================== */

/* Convert a single f16 (IEEE 754 half-precision) to f32.
 * AArch64 has native __fp16; we use memcpy to avoid strict-aliasing issues. */
static inline float f16_to_f32(uint16_t bits) {
    __fp16 h;
    memcpy(&h, &bits, sizeof(h));
    return (float)h;
}

void nomic_dequantize_q4_neon(float *out, const uint8_t *packed,
                               const uint16_t *scales_f16,
                               int n, int block_size) {
    int num_blocks = n / block_size;
    int bytes_per_block = block_size / 2;
    int8x16_t zero_point = vdupq_n_s8(8);

    for (int b = 0; b < num_blocks; b++) {
        float scale = f16_to_f32(scales_f16[b]);
        float32x4_t sv = vdupq_n_f32(scale);

        const uint8_t *src = packed + b * bytes_per_block;
        float *dst = out + b * block_size;

        /* Process 16 packed bytes (32 weights) per iteration */
        int i = 0;
        for (; i + 16 <= bytes_per_block; i += 16) {
            uint8x16_t raw = vld1q_u8(src + i);

            /* Split into low nibbles (even weights) and high nibbles (odd) */
            uint8x16_t lo = vandq_u8(raw, vdupq_n_u8(0x0F));
            uint8x16_t hi = vshrq_n_u8(raw, 4);

            /* Interleave to sequential order: [w0,w1,w2,w3,...,w31] */
            uint8x16x2_t zipped = vzipq_u8(lo, hi);

            /* --- First 16 values (zipped.val[0]) --- */
            int8x16_t s0 = vsubq_s8(vreinterpretq_s8_u8(zipped.val[0]), zero_point);

            /* Widen s8 -> s16 -> s32 -> f32, multiply by scale */
            int16x8_t s16_lo = vmovl_s8(vget_low_s8(s0));
            int16x8_t s16_hi = vmovl_s8(vget_high_s8(s0));

            int32x4_t i32_0 = vmovl_s16(vget_low_s16(s16_lo));
            int32x4_t i32_1 = vmovl_s16(vget_high_s16(s16_lo));
            int32x4_t i32_2 = vmovl_s16(vget_low_s16(s16_hi));
            int32x4_t i32_3 = vmovl_s16(vget_high_s16(s16_hi));

            int off = i * 2;
            vst1q_f32(dst + off +  0, vmulq_f32(vcvtq_f32_s32(i32_0), sv));
            vst1q_f32(dst + off +  4, vmulq_f32(vcvtq_f32_s32(i32_1), sv));
            vst1q_f32(dst + off +  8, vmulq_f32(vcvtq_f32_s32(i32_2), sv));
            vst1q_f32(dst + off + 12, vmulq_f32(vcvtq_f32_s32(i32_3), sv));

            /* --- Second 16 values (zipped.val[1]) --- */
            int8x16_t s1 = vsubq_s8(vreinterpretq_s8_u8(zipped.val[1]), zero_point);

            s16_lo = vmovl_s8(vget_low_s8(s1));
            s16_hi = vmovl_s8(vget_high_s8(s1));

            i32_0 = vmovl_s16(vget_low_s16(s16_lo));
            i32_1 = vmovl_s16(vget_high_s16(s16_lo));
            i32_2 = vmovl_s16(vget_low_s16(s16_hi));
            i32_3 = vmovl_s16(vget_high_s16(s16_hi));

            vst1q_f32(dst + off + 16, vmulq_f32(vcvtq_f32_s32(i32_0), sv));
            vst1q_f32(dst + off + 20, vmulq_f32(vcvtq_f32_s32(i32_1), sv));
            vst1q_f32(dst + off + 24, vmulq_f32(vcvtq_f32_s32(i32_2), sv));
            vst1q_f32(dst + off + 28, vmulq_f32(vcvtq_f32_s32(i32_3), sv));
        }

        /* Scalar tail for blocks not aligned to 32 elements */
        for (; i < bytes_per_block; i++) {
            uint8_t byte = src[i];
            int idx = i * 2;
            dst[idx]     = (float)((int)(byte & 0x0F) - 8) * scale;
            dst[idx + 1] = (float)((int)(byte >> 4)   - 8) * scale;
        }
    }

    /* Handle any remainder beyond full blocks (shouldn't happen with
     * properly quantized weights, but defensive) */
    int done = num_blocks * block_size;
    if (done < n) {
        float scale = (num_blocks < n / block_size + 1)
                      ? f16_to_f32(scales_f16[num_blocks]) : 0.0f;
        const uint8_t *src = packed + done / 2;
        for (int i = 0; i < n - done; i += 2) {
            uint8_t byte = src[i / 2];
            out[done + i] = (float)((int)(byte & 0x0F) - 8) * scale;
            if (done + i + 1 < n) {
                out[done + i + 1] = (float)((int)(byte >> 4) - 8) * scale;
            }
        }
    }
}

/* ========================================================================
 * Mean Pooling Kernel (NEON)
 *
 * For each sequence in the batch:
 *   1. Sum all token hidden states in the sequence
 *   2. Divide by sequence length
 *   3. Write the 768-dim (or hidden-dim) result
 *
 * Uses 8-wide NEON accumulation with scalar tail.
 * ======================================================================== */

void nomic_mean_pool_neon(float *out, const float *hidden_states,
                          const int *seq_starts, const int *seq_lens,
                          int num_seqs, int hidden) {
    for (int s = 0; s < num_seqs; s++) {
        int start = seq_starts[s];
        int len = seq_lens[s];
        float *o = out + s * hidden;

        /* Zero the output row */
        memset(o, 0, hidden * sizeof(float));

        /* Accumulate all tokens in the sequence */
        for (int t = 0; t < len; t++) {
            const float *tok = hidden_states + (start + t) * hidden;
            int d = 0;
            for (; d + 8 <= hidden; d += 8) {
                float32x4_t a0 = vld1q_f32(o + d);
                float32x4_t a1 = vld1q_f32(o + d + 4);
                float32x4_t b0 = vld1q_f32(tok + d);
                float32x4_t b1 = vld1q_f32(tok + d + 4);
                vst1q_f32(o + d,     vaddq_f32(a0, b0));
                vst1q_f32(o + d + 4, vaddq_f32(a1, b1));
            }
            for (; d < hidden; d++) {
                o[d] += tok[d];
            }
        }

        /* Divide by sequence length */
        if (len > 0) {
            float inv_len = 1.0f / (float)len;
            float32x4_t scale = vdupq_n_f32(inv_len);
            int d = 0;
            for (; d + 8 <= hidden; d += 8) {
                vst1q_f32(o + d,     vmulq_f32(vld1q_f32(o + d),     scale));
                vst1q_f32(o + d + 4, vmulq_f32(vld1q_f32(o + d + 4), scale));
            }
            for (; d < hidden; d++) {
                o[d] *= inv_len;
            }
        }
    }
}

#else /* !__ARM_NEON — scalar fallback for non-ARM builds */

void nomic_dequantize_q4_neon(float *out, const uint8_t *packed,
                               const uint16_t *scales_f16,
                               int n, int block_size) {
    int num_blocks = n / block_size;
    int bytes_per_block = block_size / 2;

    for (int b = 0; b < num_blocks; b++) {
        /* f16 to f32 scalar: decode IEEE 754 half-precision */
        uint16_t h = scales_f16[b];
        uint32_t sign = (uint32_t)(h >> 15) << 31;
        uint32_t exp  = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        uint32_t f32_bits;
        if (exp == 0) {
            f32_bits = sign; /* zero / subnormal -> zero */
        } else if (exp == 31) {
            f32_bits = sign | 0x7F800000 | (mant << 13); /* inf/nan */
        } else {
            f32_bits = sign | ((exp + 112) << 23) | (mant << 13);
        }
        float scale;
        memcpy(&scale, &f32_bits, sizeof(float));

        const uint8_t *src = packed + b * bytes_per_block;
        float *dst = out + b * block_size;

        for (int i = 0; i < bytes_per_block; i++) {
            uint8_t byte = src[i];
            dst[i * 2]     = (float)((int)(byte & 0x0F) - 8) * scale;
            dst[i * 2 + 1] = (float)((int)(byte >> 4)   - 8) * scale;
        }
    }
}

void nomic_mean_pool_neon(float *out, const float *hidden_states,
                          const int *seq_starts, const int *seq_lens,
                          int num_seqs, int hidden) {
    for (int s = 0; s < num_seqs; s++) {
        int start = seq_starts[s];
        int len = seq_lens[s];
        float *o = out + s * hidden;

        memset(o, 0, hidden * sizeof(float));
        for (int t = 0; t < len; t++) {
            const float *tok = hidden_states + (start + t) * hidden;
            for (int d = 0; d < hidden; d++) {
                o[d] += tok[d];
            }
        }
        if (len > 0) {
            float inv_len = 1.0f / (float)len;
            for (int d = 0; d < hidden; d++) {
                o[d] *= inv_len;
            }
        }
    }
}

#endif /* __ARM_NEON */
