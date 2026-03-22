/*
 * nomic_embed.c - Batched embedding inference for nomic-embed-text
 *
 * Architecture: BERT-variant encoder with bidirectional attention, NeoX RoPE,
 * SwiGLU FFN, and RMSNorm. Weights are Q4 packed with f16 block scales.
 *
 * Forward pass per layer:
 *   RMSNorm -> Q/K/V (Q4 dequant + BLAS matmul) -> RoPE -> bidirectional
 *   attention (reusing qwen windowed attention with one window per sequence)
 *   -> output projection -> residual -> RMSNorm -> SwiGLU FFN (gate/up/down
 *   all Q4) -> residual
 *
 * After all layers: final RMSNorm -> mean pooling -> output embeddings.
 *
 * Zero dynamic allocations in the forward pass. All buffers are pre-sized
 * during nomic_load based on max_batch_size * max_seq_len.
 */

#include "nomic_embed.h"
#include "qwen_asr_kernels.h"
#include "qwen_asr_safetensors.h"
#include "qwen_asr_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================
 * Weight Loading Helpers
 * ======================================================================== */

static float *load_f32_weight(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "nomic: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_f32(sf, t);
}

/*
 * Load a Q4 weight pair from safetensors:
 *   "{base_name}.weight" -> packed uint8 nibbles (mmap'd, no copy)
 *   "{base_name}.scales" -> f16 block scales (mmap'd, no copy)
 */
static int load_q4_weight(nomic_q4_weight_t *w, multi_safetensors_t *ms,
                           const char *base_name, int rows, int cols,
                           int block_size) {
    char name[512];
    safetensors_file_t *sf;
    const safetensor_t *t;

    snprintf(name, sizeof(name), "%s.weight", base_name);
    t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "nomic: Q4 packed not found: %s\n", name);
        return -1;
    }
    w->packed = (const uint8_t *)safetensors_data(sf, t);

    snprintf(name, sizeof(name), "%s.scales", base_name);
    t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "nomic: Q4 scales not found: %s\n", name);
        return -1;
    }
    w->scales_f16 = (const uint16_t *)safetensors_data(sf, t);

    w->rows = rows;
    w->cols = cols;
    w->block_size = block_size;
    return 0;
}

/*
 * Load all weights for one transformer layer.
 *
 * Tensor naming convention (inside safetensors):
 *   layers.{i}.attn_norm.weight          f32 [hidden]
 *   layers.{i}.attn.q.{weight,scales}    Q4  [hidden, hidden]
 *   layers.{i}.attn.k.{weight,scales}    Q4  [hidden, hidden]
 *   layers.{i}.attn.v.{weight,scales}    Q4  [hidden, hidden]
 *   layers.{i}.attn.o.{weight,scales}    Q4  [hidden, hidden]
 *   layers.{i}.ffn_norm.weight           f32 [hidden]
 *   layers.{i}.ffn.gate.{weight,scales}  Q4  [intermediate, hidden]
 *   layers.{i}.ffn.up.{weight,scales}    Q4  [intermediate, hidden]
 *   layers.{i}.ffn.down.{weight,scales}  Q4  [hidden, intermediate]
 */
static int load_layer(nomic_layer_t *layer, multi_safetensors_t *ms,
                       int idx, int hidden, int inter, int block_size) {
    char name[512];
    int err = 0;

    /* RMSNorm weights (f32) */
    snprintf(name, sizeof(name), "layers.%d.attn_norm.weight", idx);
    layer->rms_attn = load_f32_weight(ms, name);
    if (!layer->rms_attn) return -1;

    snprintf(name, sizeof(name), "layers.%d.ffn_norm.weight", idx);
    layer->rms_ffn = load_f32_weight(ms, name);
    if (!layer->rms_ffn) return -1;

    /* Attention Q/K/V/O projections (Q4) */
    snprintf(name, sizeof(name), "layers.%d.attn.q", idx);
    err |= load_q4_weight(&layer->wq, ms, name, hidden, hidden, block_size);
    snprintf(name, sizeof(name), "layers.%d.attn.k", idx);
    err |= load_q4_weight(&layer->wk, ms, name, hidden, hidden, block_size);
    snprintf(name, sizeof(name), "layers.%d.attn.v", idx);
    err |= load_q4_weight(&layer->wv, ms, name, hidden, hidden, block_size);
    snprintf(name, sizeof(name), "layers.%d.attn.o", idx);
    err |= load_q4_weight(&layer->wo, ms, name, hidden, hidden, block_size);

    /* SwiGLU FFN projections (Q4) */
    snprintf(name, sizeof(name), "layers.%d.ffn.gate", idx);
    err |= load_q4_weight(&layer->w_gate, ms, name, inter, hidden, block_size);
    snprintf(name, sizeof(name), "layers.%d.ffn.up", idx);
    err |= load_q4_weight(&layer->w_up, ms, name, inter, hidden, block_size);
    snprintf(name, sizeof(name), "layers.%d.ffn.down", idx);
    err |= load_q4_weight(&layer->w_down, ms, name, hidden, inter, block_size);

    return err ? -1 : 0;
}

/* ========================================================================
 * Q4 Linear: dequantize + BLAS matmul
 *
 * Dequantizes the entire weight matrix into a pre-allocated scratch buffer,
 * then calls the existing f32 linear kernel (which uses BLAS sgemm).
 * No dynamic allocation — scratch is sized at init for the largest weight.
 * ======================================================================== */

static void nomic_q4_linear(float *y, const float *x,
                             const nomic_q4_weight_t *w, float *scratch,
                             int seq_len, int in_dim, int out_dim) {
    /* Dequantize Q4 weights into scratch: [out_dim, in_dim] f32 */
    nomic_dequantize_q4_neon(scratch, w->packed, w->scales_f16,
                              out_dim * in_dim, w->block_size);

    /* y[seq, out] = x[seq, in] @ scratch[out, in]^T */
    qwen_linear_nobias(y, x, scratch, seq_len, in_dim, out_dim);
}

/* ========================================================================
 * Transformer Layer Forward Pass
 * ======================================================================== */

static void nomic_layer_forward(nomic_ctx_t *ctx, int layer_idx,
                                 int total_tokens,
                                 const int *window_starts, int n_windows) {
    nomic_layer_t *l = &ctx->layers[layer_idx];
    int hidden = ctx->hidden_dim;
    int n_heads = ctx->num_heads;
    int head_dim = ctx->head_dim;
    int inter = ctx->intermediate_dim;

    /* --- Self-Attention Block --- */

    /* Pre-attention RMSNorm */
    qwen_rms_norm(ctx->x_norm, ctx->x, l->rms_attn,
                  total_tokens, hidden, ctx->rms_eps);

    /* Q/K/V projections: dequant Q4 weights + matmul */
    nomic_q4_linear(ctx->q, ctx->x_norm, &l->wq, ctx->dequant_scratch,
                    total_tokens, hidden, hidden);
    nomic_q4_linear(ctx->k, ctx->x_norm, &l->wk, ctx->dequant_scratch,
                    total_tokens, hidden, hidden);
    nomic_q4_linear(ctx->v, ctx->x_norm, &l->wv, ctx->dequant_scratch,
                    total_tokens, hidden, hidden);

    /* Apply RoPE to Q and K */
    qwen_apply_rope_neox(ctx->q, ctx->rope_cos, ctx->rope_sin,
                         total_tokens, n_heads, head_dim);
    qwen_apply_rope_neox(ctx->k, ctx->rope_cos, ctx->rope_sin,
                         total_tokens, n_heads, head_dim);

    /* Bidirectional attention (reuse qwen windowed attention).
     * Each sequence in the batch is one window — tokens attend to all
     * other tokens within their sequence, but not across sequences. */
    float scale = 1.0f / sqrtf((float)head_dim);
    qwen_bidirectional_attention(ctx->attn_out, ctx->q, ctx->k, ctx->v,
                                 total_tokens, n_heads, head_dim, scale,
                                 window_starts, n_windows);

    /* Output projection + residual */
    nomic_q4_linear(ctx->x_norm, ctx->attn_out, &l->wo, ctx->dequant_scratch,
                    total_tokens, hidden, hidden);
    qwen_add_inplace(ctx->x, ctx->x_norm, total_tokens * hidden);

    /* --- SwiGLU FFN Block --- */

    /* Pre-FFN RMSNorm */
    qwen_rms_norm(ctx->x_norm, ctx->x, l->rms_ffn,
                  total_tokens, hidden, ctx->rms_eps);

    /* Gate and Up projections */
    nomic_q4_linear(ctx->gate, ctx->x_norm, &l->w_gate, ctx->dequant_scratch,
                    total_tokens, hidden, inter);
    nomic_q4_linear(ctx->up, ctx->x_norm, &l->w_up, ctx->dequant_scratch,
                    total_tokens, hidden, inter);

    /* SiLU(gate) * up — reuse existing activation kernels */
    qwen_silu(ctx->gate, total_tokens * inter);
    qwen_mul_inplace(ctx->gate, ctx->up, total_tokens * inter);

    /* Down projection + residual */
    nomic_q4_linear(ctx->x_norm, ctx->gate, &l->w_down, ctx->dequant_scratch,
                    total_tokens, inter, hidden);
    qwen_add_inplace(ctx->x, ctx->x_norm, total_tokens * hidden);
}

/* ========================================================================
 * nomic_load — Initialize context, load weights, allocate buffers
 * ======================================================================== */

nomic_ctx_t *nomic_load(const char *model_dir, int max_batch_size,
                         int max_seq_len) {
    nomic_ctx_t *ctx = (nomic_ctx_t *)calloc(1, sizeof(nomic_ctx_t));
    if (!ctx) return NULL;

    /* Set model dimensions (nomic-embed-text defaults) */
    ctx->hidden_dim      = NOMIC_HIDDEN_DIM;
    ctx->num_heads       = NOMIC_NUM_HEADS;
    ctx->head_dim        = NOMIC_HEAD_DIM;
    ctx->num_layers      = NOMIC_NUM_LAYERS;
    ctx->intermediate_dim = NOMIC_INTERMEDIATE;
    ctx->vocab_size      = NOMIC_VOCAB_SIZE;
    ctx->max_seq_len     = max_seq_len;
    ctx->max_batch_size  = max_batch_size;
    ctx->max_total_tokens = max_batch_size * max_seq_len;
    ctx->rope_theta      = NOMIC_ROPE_THETA;
    ctx->rms_eps         = NOMIC_RMS_EPS;
    ctx->q4_block_size   = NOMIC_Q4_BLOCK_SIZE;

    int mt     = ctx->max_total_tokens;
    int hidden = ctx->hidden_dim;
    int inter  = ctx->intermediate_dim;

    fprintf(stderr, "nomic: loading model from %s\n", model_dir);
    fprintf(stderr, "nomic: hidden=%d heads=%d layers=%d intermediate=%d\n",
            hidden, ctx->num_heads, ctx->num_layers, inter);
    fprintf(stderr, "nomic: max_batch=%d max_seq=%d max_total=%d\n",
            max_batch_size, max_seq_len, mt);

    /* Open safetensors (memory-mapped, MAP_SHARED) */
    multi_safetensors_t *ms = multi_safetensors_open(model_dir);
    if (!ms) {
        fprintf(stderr, "nomic: failed to open safetensors in %s\n", model_dir);
        free(ctx);
        return NULL;
    }
    ctx->safetensors = ms;

    /* Load tokenizer */
    char vocab_path[1024];
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", model_dir);
    ctx->tokenizer = qwen_tokenizer_load(vocab_path);
    if (!ctx->tokenizer) {
        fprintf(stderr, "nomic: failed to load tokenizer from %s\n", vocab_path);
        multi_safetensors_close(ms);
        free(ctx);
        return NULL;
    }

    /* Load token embeddings (f32, [vocab_size, hidden]) */
    ctx->tok_embeddings = load_f32_weight(ms, "embed.weight");
    if (!ctx->tok_embeddings) goto fail;

    /* Load transformer layers */
    ctx->layers = (nomic_layer_t *)calloc(ctx->num_layers, sizeof(nomic_layer_t));
    if (!ctx->layers) goto fail;

    for (int i = 0; i < ctx->num_layers; i++) {
        if (load_layer(&ctx->layers[i], ms, i, hidden, inter,
                       ctx->q4_block_size) != 0) {
            fprintf(stderr, "nomic: failed to load layer %d\n", i);
            goto fail;
        }
    }

    /* Load final RMSNorm weight */
    ctx->final_norm_weight = load_f32_weight(ms, "final_norm.weight");
    if (!ctx->final_norm_weight) goto fail;

    /* ---- Allocate activation buffers (all sized for worst case) ---- */

    ctx->x         = (float *)malloc((size_t)mt * hidden * sizeof(float));
    ctx->x_norm    = (float *)malloc((size_t)mt * hidden * sizeof(float));
    ctx->q         = (float *)malloc((size_t)mt * hidden * sizeof(float));
    ctx->k         = (float *)malloc((size_t)mt * hidden * sizeof(float));
    ctx->v         = (float *)malloc((size_t)mt * hidden * sizeof(float));
    ctx->attn_out  = (float *)malloc((size_t)mt * hidden * sizeof(float));
    ctx->gate      = (float *)malloc((size_t)mt * inter * sizeof(float));
    ctx->up        = (float *)malloc((size_t)mt * inter * sizeof(float));

    /* Dequant scratch: sized for the largest weight matrix.
     * Largest is gate or up: [intermediate, hidden] = inter * hidden */
    size_t max_weight = (size_t)inter * hidden;
    ctx->dequant_scratch = (float *)malloc(max_weight * sizeof(float));

    /* Integer / position buffers */
    ctx->token_ids     = (int *)malloc((size_t)mt * sizeof(int));
    ctx->seq_lens      = (int *)malloc((size_t)max_batch_size * sizeof(int));
    ctx->seq_starts    = (int *)malloc((size_t)max_batch_size * sizeof(int));
    ctx->window_starts = (int *)malloc((size_t)(max_batch_size + 1) * sizeof(int));
    ctx->positions     = (int *)malloc((size_t)mt * sizeof(int));
    ctx->rope_cos      = (float *)malloc((size_t)mt * ctx->head_dim * sizeof(float));
    ctx->rope_sin      = (float *)malloc((size_t)mt * ctx->head_dim * sizeof(float));

    /* Verify all allocations succeeded */
    if (!ctx->x || !ctx->x_norm || !ctx->q || !ctx->k || !ctx->v ||
        !ctx->attn_out || !ctx->gate || !ctx->up || !ctx->dequant_scratch ||
        !ctx->token_ids || !ctx->seq_lens || !ctx->seq_starts ||
        !ctx->window_starts || !ctx->positions ||
        !ctx->rope_cos || !ctx->rope_sin) {
        fprintf(stderr, "nomic: buffer allocation failed\n");
        goto fail;
    }

    fprintf(stderr, "nomic: model loaded successfully\n");
    return ctx;

fail:
    nomic_free(ctx);
    return NULL;
}

/* ========================================================================
 * nomic_free — Release all resources
 * ======================================================================== */

void nomic_free(nomic_ctx_t *ctx) {
    if (!ctx) return;

    /* Activation buffers */
    free(ctx->x);
    free(ctx->x_norm);
    free(ctx->q);
    free(ctx->k);
    free(ctx->v);
    free(ctx->attn_out);
    free(ctx->gate);
    free(ctx->up);
    free(ctx->dequant_scratch);
    free(ctx->token_ids);
    free(ctx->seq_lens);
    free(ctx->seq_starts);
    free(ctx->window_starts);
    free(ctx->positions);
    free(ctx->rope_cos);
    free(ctx->rope_sin);

    /* f32 weight copies (embedding, norms) */
    free(ctx->tok_embeddings);
    free(ctx->final_norm_weight);
    if (ctx->layers) {
        for (int i = 0; i < ctx->num_layers; i++) {
            free(ctx->layers[i].rms_attn);
            free(ctx->layers[i].rms_ffn);
        }
        free(ctx->layers);
    }

    /* Q4 weight data lives in mmap'd safetensors — freed by close */
    if (ctx->safetensors)
        multi_safetensors_close((multi_safetensors_t *)ctx->safetensors);

    if (ctx->tokenizer)
        qwen_tokenizer_free((qwen_tokenizer_t *)ctx->tokenizer);

    free(ctx);
}

/* ========================================================================
 * nomic_embed_batch — Compute embeddings for a batch of strings
 *
 * Zero dynamic allocations on the forward-pass hot path.
 * (The tokenizer internally mallocs/frees per-string; this cost is
 * negligible compared to the transformer compute.)
 * ======================================================================== */

void nomic_embed_batch(nomic_ctx_t *ctx, const char **strings,
                       int num_strings, float *out_embeddings) {
    if (num_strings <= 0) return;
    if (num_strings > ctx->max_batch_size) {
        fprintf(stderr, "nomic: batch size %d exceeds max %d\n",
                num_strings, ctx->max_batch_size);
        return;
    }

    int hidden   = ctx->hidden_dim;
    int head_dim = ctx->head_dim;

    /* ---- Phase 1: Tokenize all strings ---- */

    int total_tokens = 0;
    for (int i = 0; i < num_strings; i++) {
        int n_tokens = 0;
        int *ids = qwen_tokenizer_encode(
            (const qwen_tokenizer_t *)ctx->tokenizer, strings[i], &n_tokens);
        if (!ids) {
            fprintf(stderr, "nomic: tokenization failed for string %d\n", i);
            ctx->seq_starts[i] = total_tokens;
            ctx->seq_lens[i] = 0;
            continue;
        }

        /* Truncate to max_seq_len */
        if (n_tokens > ctx->max_seq_len) n_tokens = ctx->max_seq_len;

        /* Bounds check: don't exceed pre-allocated buffer */
        if (total_tokens + n_tokens > ctx->max_total_tokens) {
            fprintf(stderr, "nomic: total tokens exceed buffer, truncating\n");
            n_tokens = ctx->max_total_tokens - total_tokens;
            if (n_tokens <= 0) { free(ids); break; }
        }

        ctx->seq_starts[i] = total_tokens;
        ctx->seq_lens[i] = n_tokens;
        memcpy(ctx->token_ids + total_tokens, ids, n_tokens * sizeof(int));
        total_tokens += n_tokens;
        free(ids);
    }

    if (total_tokens == 0) return;

    /* ---- Phase 2: Build attention windows and position IDs ---- */

    /* One window per sequence: tokens attend within their own sequence */
    for (int i = 0; i < num_strings; i++) {
        ctx->window_starts[i] = ctx->seq_starts[i];
    }
    ctx->window_starts[num_strings] = total_tokens;

    /* Position IDs: 0, 1, 2, ... per sequence (reset at each boundary) */
    for (int i = 0; i < num_strings; i++) {
        int start = ctx->seq_starts[i];
        int len = ctx->seq_lens[i];
        for (int t = 0; t < len; t++) {
            ctx->positions[start + t] = t;
        }
    }

    /* Precompute RoPE cos/sin for all token positions */
    qwen_compute_rope_neox(ctx->rope_cos, ctx->rope_sin, ctx->positions,
                            total_tokens, head_dim, ctx->rope_theta);

    /* ---- Phase 3: Token embedding lookup ---- */

    for (int t = 0; t < total_tokens; t++) {
        int id = ctx->token_ids[t];
        if (id >= 0 && id < ctx->vocab_size) {
            memcpy(ctx->x + t * hidden,
                   ctx->tok_embeddings + id * hidden,
                   hidden * sizeof(float));
        } else {
            /* Out-of-vocab: zero embedding */
            memset(ctx->x + t * hidden, 0, hidden * sizeof(float));
        }
    }

    /* ---- Phase 4: Transformer layers ---- */

    for (int l = 0; l < ctx->num_layers; l++) {
        nomic_layer_forward(ctx, l, total_tokens,
                            ctx->window_starts, num_strings);
    }

    /* ---- Phase 5: Final RMSNorm ---- */

    qwen_rms_norm(ctx->x, ctx->x, ctx->final_norm_weight,
                  total_tokens, hidden, ctx->rms_eps);

    /* ---- Phase 6: Mean pooling ---- */

    nomic_mean_pool_neon(out_embeddings, ctx->x,
                         ctx->seq_starts, ctx->seq_lens,
                         num_strings, hidden);
}
