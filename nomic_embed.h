/*
 * nomic_embed.h - Batched embedding inference for nomic-embed-text
 *
 * Pure C engine with Q4 quantized weights, designed for FFI from a Rust host.
 * All activation buffers are pre-allocated at load time; nomic_embed_batch
 * performs zero dynamic allocations in the forward pass.
 *
 * Architecture: BERT-variant encoder (bidirectional attention, RoPE, SwiGLU,
 * RMSNorm, mean pooling). Weights are 4-bit packed with f16 block scales.
 */

#ifndef NOMIC_EMBED_H
#define NOMIC_EMBED_H

#include <stdint.h>

/* ========================================================================
 * Model Constants (nomic-embed-text defaults)
 * ======================================================================== */

#define NOMIC_HIDDEN_DIM      768
#define NOMIC_NUM_HEADS       12
#define NOMIC_HEAD_DIM        64
#define NOMIC_NUM_LAYERS      12
#define NOMIC_INTERMEDIATE    3072
#define NOMIC_VOCAB_SIZE      30528
#define NOMIC_ROPE_THETA      10000.0f
#define NOMIC_RMS_EPS         1e-5f
#define NOMIC_Q4_BLOCK_SIZE   32

/* ========================================================================
 * Q4 Weight Descriptor
 *
 * Custom 4-bit packed integer format inside standard .safetensors files.
 * Each byte holds two 4-bit weights (low nibble = even index, high = odd).
 * Block scales are stored as a separate f16 tensor: one scale per block_size
 * elements. Dequantization: f32_val = (nibble - 8) * scale
 * ======================================================================== */

typedef struct {
    const uint8_t  *packed;      /* [rows * cols / 2] packed nibble pairs */
    const uint16_t *scales_f16;  /* [rows * cols / block_size] f16 scales */
    int rows;
    int cols;
    int block_size;
} nomic_q4_weight_t;

/* ========================================================================
 * Per-Layer Weights
 * ======================================================================== */

typedef struct {
    /* Self-attention projections (all [hidden, hidden]) */
    nomic_q4_weight_t wq;
    nomic_q4_weight_t wk;
    nomic_q4_weight_t wv;
    nomic_q4_weight_t wo;
    float *rms_attn;             /* [hidden] pre-attention RMSNorm weight */

    /* SwiGLU FFN */
    nomic_q4_weight_t w_gate;    /* [intermediate, hidden] */
    nomic_q4_weight_t w_up;      /* [intermediate, hidden] */
    nomic_q4_weight_t w_down;    /* [hidden, intermediate] */
    float *rms_ffn;              /* [hidden] pre-FFN RMSNorm weight */
} nomic_layer_t;

/* ========================================================================
 * Execution Context
 *
 * Holds model weights (mmap'd Q4 pointers + f32 norms/embeddings) and all
 * pre-allocated activation buffers sized for max_batch_size * max_seq_len.
 * ======================================================================== */

typedef struct {
    /* Model dimensions */
    int hidden_dim;
    int num_heads;
    int head_dim;
    int num_layers;
    int intermediate_dim;
    int vocab_size;
    int max_seq_len;
    int max_batch_size;
    int max_total_tokens;        /* max_batch_size * max_seq_len */
    float rope_theta;
    float rms_eps;
    int q4_block_size;

    /* Model weights */
    float *tok_embeddings;       /* [vocab_size, hidden] f32 */
    nomic_layer_t *layers;       /* [num_layers] */
    float *final_norm_weight;    /* [hidden] f32 */

    /* Safetensors handle (kept open for mmap'd Q4 pointers) */
    void *safetensors;

    /* Tokenizer handle */
    void *tokenizer;

    /* --- Pre-allocated activation buffers (hot path, zero malloc) --- */

    float *x;                    /* [max_total, hidden] hidden state */
    float *x_norm;               /* [max_total, hidden] norm scratch */
    float *q;                    /* [max_total, hidden] queries */
    float *k;                    /* [max_total, hidden] keys */
    float *v;                    /* [max_total, hidden] values */
    float *attn_out;             /* [max_total, hidden] attention output */
    float *gate;                 /* [max_total, intermediate] gate proj */
    float *up;                   /* [max_total, intermediate] up proj */
    float *dequant_scratch;      /* [intermediate * hidden] dequant buffer */

    /* Integer / position buffers */
    int *token_ids;              /* [max_total] tokenized input */
    int *seq_lens;               /* [max_batch] per-sequence lengths */
    int *seq_starts;             /* [max_batch] token offset per sequence */
    int *window_starts;          /* [max_batch + 1] attention window bounds */
    int *positions;              /* [max_total] per-token position IDs */
    float *rope_cos;             /* [max_total, head_dim] RoPE cosines */
    float *rope_sin;             /* [max_total, head_dim] RoPE sines */
} nomic_ctx_t;

/* ========================================================================
 * FFI API
 * ======================================================================== */

/*
 * Load model from directory containing .safetensors and vocab.json.
 * Pre-allocates all buffers for the given batch/seq limits.
 * Returns NULL on failure.
 */
nomic_ctx_t *nomic_load(const char *model_dir, int max_batch_size, int max_seq_len);

/*
 * Free all resources (weights, buffers, safetensors mmap, tokenizer).
 */
void nomic_free(nomic_ctx_t *ctx);

/*
 * Compute embeddings for a batch of UTF-8 strings.
 *
 * out_embeddings must be caller-allocated: [num_strings * hidden_dim] floats.
 * num_strings must be <= max_batch_size passed to nomic_load.
 * Strings longer than max_seq_len are truncated.
 *
 * This function performs ZERO dynamic allocations (except inside the
 * tokenizer, which is not on the compute-critical path).
 */
void nomic_embed_batch(nomic_ctx_t *ctx, const char **strings,
                       int num_strings, float *out_embeddings);

/* ========================================================================
 * NEON Kernels (nomic_kernels_neon.c)
 * ======================================================================== */

/*
 * Dequantize Q4 packed weights to f32.
 *
 * packed:      [n/2] bytes, each holding two 4-bit weights
 * scales_f16:  [n/block_size] f16 per-block scales
 * out:         [n] f32 output
 * n:           total number of elements (must be multiple of block_size)
 * block_size:  elements per scale block
 *
 * Dequant formula: out[i] = (nibble_i - 8) * scale[i / block_size]
 */
void nomic_dequantize_q4_neon(float *out, const uint8_t *packed,
                               const uint16_t *scales_f16,
                               int n, int block_size);

/*
 * Mean pool hidden states across the token dimension.
 *
 * hidden_states: [total_tokens, hidden] row-major
 * seq_starts:    [num_seqs] start index of each sequence
 * seq_lens:      [num_seqs] length of each sequence
 * out:           [num_seqs, hidden] output embeddings
 * num_seqs:      number of sequences
 * hidden:        hidden dimension
 */
void nomic_mean_pool_neon(float *out, const float *hidden_states,
                          const int *seq_starts, const int *seq_lens,
                          int num_seqs, int hidden);

#endif /* NOMIC_EMBED_H */
