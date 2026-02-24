/*
 * qwen_asr_decoder.c - Qwen3 LLM decoder
 *
 * Architecture (per layer):
 *   RMSNorm -> QKV (no bias) -> per-head Q/K RMSNorm -> NeoX RoPE
 *   -> Causal GQA attention -> Output proj -> residual
 *   RMSNorm -> SwiGLU MLP (gate/up/down, no bias) -> residual
 *
 * MoE variant (30B): replaces dense SwiGLU MLP with router + 128 experts.
 * Expert bf16 pointers pre-resolved at load time from mmap'd safetensors.
 *
 * Features: Q/K per-head RMSNorm, NeoX split-half RoPE, GQA,
 * tied embeddings (tok_embeddings == lm_head).
 */

#include "qwen_asr.h"
#include "qwen_asr_kernels.h"
#include "qwen_asr_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <unistd.h>

/* ========================================================================
 * Weight Loading
 * ======================================================================== */

static float *load_f32(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "decoder: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_f32(sf, t);
}

static uint16_t *load_bf16_direct(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "decoder: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_bf16_direct(sf, t);
}

int qwen_decoder_load(qwen_decoder_t *dec, multi_safetensors_t *ms,
                       const qwen_config_t *cfg) {
    char name[512];

    /* Token embeddings (large, bf16 mmap direct) */
    dec->tok_embeddings_bf16 = load_bf16_direct(ms,
        "thinker.model.embed_tokens.weight");
    if (!dec->tok_embeddings_bf16) return -1;

    /* Transformer layers */
    for (int i = 0; i < cfg->dec_layers; i++) {
        qwen_dec_layer_t *l = &dec->layers[i];

        /* Attention weights (bf16, no bias) */
        snprintf(name, sizeof(name), "thinker.model.layers.%d.self_attn.q_proj.weight", i);
        l->wq_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "thinker.model.layers.%d.self_attn.k_proj.weight", i);
        l->wk_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "thinker.model.layers.%d.self_attn.v_proj.weight", i);
        l->wv_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "thinker.model.layers.%d.self_attn.o_proj.weight", i);
        l->wo_weight_bf16 = load_bf16_direct(ms, name);

        /* Per-head Q/K RMSNorm weights */
        snprintf(name, sizeof(name), "thinker.model.layers.%d.self_attn.q_norm.weight", i);
        l->q_norm_weight = load_f32(ms, name);
        snprintf(name, sizeof(name), "thinker.model.layers.%d.self_attn.k_norm.weight", i);
        l->k_norm_weight = load_f32(ms, name);

        /* RMSNorm weights */
        snprintf(name, sizeof(name), "thinker.model.layers.%d.input_layernorm.weight", i);
        l->input_norm = load_f32(ms, name);
        snprintf(name, sizeof(name), "thinker.model.layers.%d.post_attention_layernorm.weight", i);
        l->post_attn_norm = load_f32(ms, name);

        if (!l->wq_weight_bf16 || !l->wk_weight_bf16 ||
            !l->wv_weight_bf16 || !l->wo_weight_bf16) {
            fprintf(stderr, "decoder: failed to load attn weights for layer %d\n", i);
            return -1;
        }

        if (cfg->is_moe) {
            /* MoE layer: load router gate weight as f32 */
            snprintf(name, sizeof(name), "thinker.model.layers.%d.mlp.gate.weight", i);
            l->moe_gate_weight = load_f32(ms, name);
            if (!l->moe_gate_weight) {
                fprintf(stderr, "decoder: failed to load MoE gate for layer %d\n", i);
                return -1;
            }
            /* Pre-resolve all expert bf16 pointers at load time */
            int ne = cfg->num_experts;
            l->moe_experts = calloc(ne, sizeof(*l->moe_experts));
            if (!l->moe_experts) {
                fprintf(stderr, "decoder: alloc moe_experts failed layer %d\n", i);
                return -1;
            }
            for (int e = 0; e < ne; e++) {
                snprintf(name, sizeof(name),
                         "thinker.model.layers.%d.mlp.experts.%d.gate_proj.weight", i, e);
                l->moe_experts[e].gate_proj = load_bf16_direct(ms, name);
                snprintf(name, sizeof(name),
                         "thinker.model.layers.%d.mlp.experts.%d.up_proj.weight", i, e);
                l->moe_experts[e].up_proj = load_bf16_direct(ms, name);
                snprintf(name, sizeof(name),
                         "thinker.model.layers.%d.mlp.experts.%d.down_proj.weight", i, e);
                l->moe_experts[e].down_proj = load_bf16_direct(ms, name);
                if (!l->moe_experts[e].gate_proj || !l->moe_experts[e].up_proj ||
                    !l->moe_experts[e].down_proj) {
                    fprintf(stderr, "decoder: failed to load expert %d weights for layer %d\n", e, i);
                    return -1;
                }
            }
            l->gate_weight_bf16 = NULL;
            l->up_weight_bf16 = NULL;
            l->down_weight_bf16 = NULL;
            l->gate_up_fused_bf16 = NULL;
        } else {
            /* Dense MLP: load gate/up/down and fuse */
            snprintf(name, sizeof(name), "thinker.model.layers.%d.mlp.gate_proj.weight", i);
            l->gate_weight_bf16 = load_bf16_direct(ms, name);
            snprintf(name, sizeof(name), "thinker.model.layers.%d.mlp.up_proj.weight", i);
            l->up_weight_bf16 = load_bf16_direct(ms, name);
            snprintf(name, sizeof(name), "thinker.model.layers.%d.mlp.down_proj.weight", i);
            l->down_weight_bf16 = load_bf16_direct(ms, name);

            if (!l->gate_weight_bf16 || !l->up_weight_bf16 || !l->down_weight_bf16) {
                fprintf(stderr, "decoder: failed to load MLP weights for layer %d\n", i);
                return -1;
            }

            /* Fuse gate+up weights: interleave rows [gate_row0, up_row0, gate_row1, up_row1, ...] */
            {
                int inter = cfg->dec_intermediate;
                int hidden = cfg->dec_hidden;
                size_t row_bytes = (size_t)hidden * sizeof(uint16_t);
                l->gate_up_fused_bf16 = (uint16_t *)malloc(2 * (size_t)inter * row_bytes);
                for (int r = 0; r < inter; r++) {
                    memcpy(l->gate_up_fused_bf16 + (size_t)(2 * r) * hidden,
                           l->gate_weight_bf16 + (size_t)r * hidden, row_bytes);
                    memcpy(l->gate_up_fused_bf16 + (size_t)(2 * r + 1) * hidden,
                           l->up_weight_bf16 + (size_t)r * hidden, row_bytes);
                }
            }
            l->moe_gate_weight = NULL;
        }
    }

    /* Final RMSNorm */
    dec->norm = load_f32(ms, "thinker.model.norm.weight");
    if (!dec->norm) return -1;

    return 0;
}

/* ========================================================================
 * KV Cache Management
 * ======================================================================== */

static int kv_cache_init(qwen_ctx_t *ctx, int max_seq) {
    int kv_dim = ctx->config.dec_kv_heads * ctx->config.dec_head_dim;
    size_t cache_size = (size_t)ctx->config.dec_layers * max_seq * kv_dim * sizeof(float);
    ctx->kv_cache_k = (float *)calloc(1, cache_size);
    ctx->kv_cache_v = (float *)calloc(1, cache_size);
    ctx->kv_cache_len = 0;
    ctx->kv_cache_max = max_seq;
    if (!ctx->kv_cache_k || !ctx->kv_cache_v) return -1;
    return 0;
}

static int kv_cache_grow(qwen_ctx_t *ctx, int required) {
    if (required <= ctx->kv_cache_max) return 0;

    int kv_dim = ctx->config.dec_kv_heads * ctx->config.dec_head_dim;
    int new_max = ctx->kv_cache_max;
    while (new_max < required) new_max *= 2;

    size_t new_stride = (size_t)new_max * kv_dim;
    size_t old_stride = (size_t)ctx->kv_cache_max * kv_dim;
    size_t total = (size_t)ctx->config.dec_layers * new_stride * sizeof(float);

    float *new_k = (float *)calloc(1, total);
    float *new_v = (float *)calloc(1, total);
    if (!new_k || !new_v) { free(new_k); free(new_v); return -1; }

    size_t copy = (size_t)ctx->kv_cache_len * kv_dim * sizeof(float);
    for (int l = 0; l < ctx->config.dec_layers; l++) {
        memcpy(new_k + l * new_stride, ctx->kv_cache_k + l * old_stride, copy);
        memcpy(new_v + l * new_stride, ctx->kv_cache_v + l * old_stride, copy);
    }

    free(ctx->kv_cache_k);
    free(ctx->kv_cache_v);
    ctx->kv_cache_k = new_k;
    ctx->kv_cache_v = new_v;
    ctx->kv_cache_max = new_max;
    return 0;
}

static float *kv_cache_k_at(qwen_ctx_t *ctx, int layer, int pos) {
    int kv_dim = ctx->config.dec_kv_heads * ctx->config.dec_head_dim;
    return ctx->kv_cache_k + ((size_t)layer * ctx->kv_cache_max + pos) * kv_dim;
}

static float *kv_cache_v_at(qwen_ctx_t *ctx, int layer, int pos) {
    int kv_dim = ctx->config.dec_kv_heads * ctx->config.dec_head_dim;
    return ctx->kv_cache_v + ((size_t)layer * ctx->kv_cache_max + pos) * kv_dim;
}

static int ensure_prefill_buffers(qwen_ctx_t *ctx, int seq_len) {
    const qwen_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int q_dim = cfg->dec_heads * cfg->dec_head_dim;
    int kv_dim = cfg->dec_kv_heads * cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;

    if (seq_len <= ctx->pref_seq_cap) return 0;

    int new_cap = ctx->pref_seq_cap > 0 ? ctx->pref_seq_cap : 64;
    while (new_cap < seq_len) new_cap *= 2;

#define REALLOC_PREF(ptr, count) do {                                          \
    void *tmp__ = realloc((ptr), (size_t)(count) * sizeof(float));             \
    if (!tmp__) return -1;                                                      \
    (ptr) = (float *)tmp__;                                                     \
} while (0)

    REALLOC_PREF(ctx->pref_x, new_cap * dim);
    REALLOC_PREF(ctx->pref_x_norm, new_cap * dim);
    REALLOC_PREF(ctx->pref_q, new_cap * q_dim);
    REALLOC_PREF(ctx->pref_k, new_cap * kv_dim);
    REALLOC_PREF(ctx->pref_v, new_cap * kv_dim);
    REALLOC_PREF(ctx->pref_attn_out, new_cap * q_dim);
    REALLOC_PREF(ctx->pref_proj_out, new_cap * dim);
    REALLOC_PREF(ctx->pref_ffn_out, new_cap * dim);
    REALLOC_PREF(ctx->pref_gate, new_cap * intermediate);
    REALLOC_PREF(ctx->pref_gate_up, new_cap * 2 * intermediate);

#undef REALLOC_PREF

    ctx->pref_seq_cap = new_cap;
    return 0;
}

static int ensure_rope_inv_freq(qwen_ctx_t *ctx, int head_dim, float theta) {
    int half = head_dim / 2;
    if (ctx->rope_inv_freq && ctx->rope_inv_freq_half == half) return 0;

    float *inv = (float *)realloc(ctx->rope_inv_freq, (size_t)half * sizeof(float));
    if (!inv) return -1;
    ctx->rope_inv_freq = inv;

    for (int d = 0; d < half; d++) {
        ctx->rope_inv_freq[d] = 1.0f / powf(theta, (float)(2 * d) / (float)head_dim);
    }
    ctx->rope_inv_freq_half = half;
    return 0;
}

static int ensure_rope_cache(qwen_ctx_t *ctx, int required_pos, int head_dim, float theta) {
    if (required_pos <= ctx->rope_cache_cap) return 0;
    if (ensure_rope_inv_freq(ctx, head_dim, theta) != 0) return -1;

    int new_cap = ctx->rope_cache_cap > 0 ? ctx->rope_cache_cap : 1024;
    while (new_cap < required_pos) new_cap *= 2;

    size_t n = (size_t)new_cap * head_dim;
    float *new_cos = (float *)realloc(ctx->rope_cache_cos, n * sizeof(float));
    if (!new_cos) return -1;
    ctx->rope_cache_cos = new_cos;

    float *new_sin = (float *)realloc(ctx->rope_cache_sin, n * sizeof(float));
    if (!new_sin) return -1;
    ctx->rope_cache_sin = new_sin;

    int half = head_dim / 2;
    for (int pos = ctx->rope_cache_cap; pos < new_cap; pos++) {
        float p = (float)pos;
        float *cos_row = ctx->rope_cache_cos + (size_t)pos * head_dim;
        float *sin_row = ctx->rope_cache_sin + (size_t)pos * head_dim;
        for (int d = 0; d < half; d++) {
            float angle = p * ctx->rope_inv_freq[d];
            float c = cosf(angle);
            float s = sinf(angle);
            cos_row[d] = c;
            cos_row[half + d] = c;
            sin_row[d] = s;
            sin_row[half + d] = s;
        }
    }

    ctx->rope_cache_cap = new_cap;
    return 0;
}

/* ========================================================================
 * MoE Forward (Single Token)
 * ======================================================================== */

static void ensure_moe_buffers(qwen_ctx_t *ctx) {
    if (ctx->moe_router_logits) return;
    const qwen_config_t *cfg = &ctx->config;
    int moe_inter = cfg->moe_intermediate;
    int dim = cfg->dec_hidden;
    ctx->moe_router_logits = (float *)malloc(cfg->num_experts * sizeof(float));
    ctx->moe_gate_buf     = (float *)malloc(moe_inter * sizeof(float));
    ctx->moe_up_buf       = (float *)malloc(moe_inter * sizeof(float));
    ctx->moe_expert_out   = (float *)malloc(dim * sizeof(float));
    ctx->moe_accum        = (float *)malloc(dim * sizeof(float));
}

/* Pre-fault all MoE expert pages into RAM via madvise(MADV_WILLNEED). */
void qwen_decoder_moe_preload(qwen_decoder_t *dec, const qwen_config_t *cfg) {
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) page_size = 4096;
    size_t total_bytes = 0;

    for (int i = 0; i < cfg->dec_layers; i++) {
        qwen_dec_layer_t *l = &dec->layers[i];
        if (!l->moe_experts) continue;

        /* Each expert has 3 weight tensors:
         *   gate_proj: [moe_intermediate, hidden] bf16
         *   up_proj:   [moe_intermediate, hidden] bf16
         *   down_proj: [hidden, moe_intermediate] bf16 */
        size_t gate_up_bytes = (size_t)cfg->moe_intermediate * cfg->dec_hidden * sizeof(uint16_t);
        size_t down_bytes    = (size_t)cfg->dec_hidden * cfg->moe_intermediate * sizeof(uint16_t);

        for (int e = 0; e < cfg->num_experts; e++) {
            uint16_t *ptrs[3] = {
                l->moe_experts[e].gate_proj,
                l->moe_experts[e].up_proj,
                l->moe_experts[e].down_proj
            };
            size_t sizes[3] = { gate_up_bytes, gate_up_bytes, down_bytes };
            for (int w = 0; w < 3; w++) {
                uintptr_t addr = (uintptr_t)ptrs[w];
                uintptr_t aligned = addr & ~((uintptr_t)page_size - 1);
                size_t len = sizes[w] + (addr - aligned);
                madvise((void *)aligned, len, MADV_WILLNEED);
                total_bytes += sizes[w];
            }
        }
    }

    fprintf(stderr, "MoE preload: advised %.1f GB of expert pages\n",
            (double)total_bytes / (1024.0 * 1024.0 * 1024.0));
}

/* Single-token MoE forward: routes x_norm to top-k experts and accumulates.
 * x_norm: [hidden] (post-attn RMSNorm output, read-only)
 * residual x: [hidden] (updated in-place: x += weighted sum of expert outputs)
 * Expert bf16 pointers are pre-resolved at load time (mmap'd safetensors). */
static void moe_forward_single(qwen_ctx_t *ctx, float *x, const float *x_norm,
                                const qwen_dec_layer_t *l) {
    const qwen_config_t *cfg = &ctx->config;
    int hidden = cfg->dec_hidden;
    int num_experts = cfg->num_experts;
    int top_k = cfg->num_experts_per_tok;
    int moe_inter = cfg->moe_intermediate;

    ensure_moe_buffers(ctx);
    float *router_logits = ctx->moe_router_logits;
    float *gate_buf = ctx->moe_gate_buf;
    float *up_buf = ctx->moe_up_buf;
    float *expert_out = ctx->moe_expert_out;
    float *accum = ctx->moe_accum;

    /* 1. Compute router logits: gate_weight @ x_norm -> [num_experts]
     *    gate_weight is [num_experts, hidden] stored row-major as f32. */
    for (int e = 0; e < num_experts; e++) {
        const float *row = l->moe_gate_weight + (size_t)e * hidden;
        float dot = 0.0f;
        for (int d = 0; d < hidden; d++) dot += row[d] * x_norm[d];
        router_logits[e] = dot;
    }

    /* 2. Find top-k expert indices (simple selection) */
    int topk_idx[8]; /* num_experts_per_tok is 8 */
    float topk_logits[8];
    for (int k = 0; k < top_k; k++) {
        int best = -1;
        float best_val = -1e30f;
        for (int e = 0; e < num_experts; e++) {
            /* Skip already-selected experts */
            int skip = 0;
            for (int j = 0; j < k; j++) {
                if (topk_idx[j] == e) { skip = 1; break; }
            }
            if (skip) continue;
            if (router_logits[e] > best_val) {
                best_val = router_logits[e];
                best = e;
            }
        }
        topk_idx[k] = best;
        topk_logits[k] = best_val;
    }

    /* 3. Softmax over top-k logits to get routing weights */
    float max_logit = topk_logits[0];
    for (int k = 1; k < top_k; k++)
        if (topk_logits[k] > max_logit) max_logit = topk_logits[k];

    float sum_exp = 0.0f;
    float topk_weights[8];
    for (int k = 0; k < top_k; k++) {
        topk_weights[k] = expf(topk_logits[k] - max_logit);
        sum_exp += topk_weights[k];
    }
    for (int k = 0; k < top_k; k++)
        topk_weights[k] /= sum_exp;

    /* norm_topk_prob: normalize so weights sum to 1 (already guaranteed by
     * softmax, but matches HF implementation). No-op here. */

    /* 4. Zero accumulator */
    memset(accum, 0, (size_t)hidden * sizeof(float));

    /* 5. For each selected expert: use pre-resolved pointers, compute SwiGLU, accumulate */
    for (int k = 0; k < top_k; k++) {
        int eidx = topk_idx[k];
        float w = topk_weights[k];

        uint16_t *gate_w = l->moe_experts[eidx].gate_proj;
        uint16_t *up_w   = l->moe_experts[eidx].up_proj;
        uint16_t *down_w = l->moe_experts[eidx].down_proj;

        /* SwiGLU: out = down(silu(gate(x_norm)) * up(x_norm)) */
        qwen_linear_nobias_bf16(gate_buf, x_norm, gate_w, 1, hidden, moe_inter);
        qwen_linear_nobias_bf16(up_buf, x_norm, up_w, 1, hidden, moe_inter);
        qwen_silu(gate_buf, moe_inter);
        qwen_mul_inplace(gate_buf, up_buf, moe_inter);
        qwen_linear_nobias_bf16(expert_out, gate_buf, down_w, 1, moe_inter, hidden);

        /* Weighted accumulate: accum += w * expert_out */
        for (int d = 0; d < hidden; d++)
            accum[d] += w * expert_out[d];
    }

    /* 6. Add MoE output to residual */
    qwen_add_inplace(x, accum, hidden);
}

/* ========================================================================
 * Decoder Prefill (Multiple Tokens)
 * ======================================================================== */

void qwen_decoder_prefill(qwen_ctx_t *ctx, const float *input_embeds, int seq_len) {
    qwen_decoder_t *dec = &ctx->decoder;
    const qwen_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int n_heads = cfg->dec_heads;
    int n_kv_heads = cfg->dec_kv_heads;
    int head_dim = cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;
    float eps = cfg->dec_rms_norm_eps;
    float theta = cfg->dec_rope_theta;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    /* Ensure KV cache */
    if (!ctx->kv_cache_k) {
        if (kv_cache_init(ctx, seq_len + 1024) != 0) return;
    } else if (ctx->kv_cache_len + seq_len > ctx->kv_cache_max) {
        if (kv_cache_grow(ctx, ctx->kv_cache_len + seq_len + 1024) != 0) return;
    }

    if (ensure_prefill_buffers(ctx, seq_len) != 0) return;

    float *x = ctx->pref_x;
    float *x_norm = ctx->pref_x_norm;
    float *q = ctx->pref_q;
    float *k = ctx->pref_k;
    float *v = ctx->pref_v;
    float *attn_out = ctx->pref_attn_out;
    float *proj_out = ctx->pref_proj_out;
    float *ffn_out = ctx->pref_ffn_out;
    float *gate = ctx->pref_gate;
    float *gate_up = ctx->pref_gate_up;

    memcpy(x, input_embeds, (size_t)seq_len * dim * sizeof(float));

    int start_pos = ctx->kv_cache_len;
    if (ensure_rope_cache(ctx, start_pos + seq_len, head_dim, theta) != 0) return;
    const float *rope_cos = ctx->rope_cache_cos + (size_t)start_pos * head_dim;
    const float *rope_sin = ctx->rope_cache_sin + (size_t)start_pos * head_dim;

    float scale = 1.0f / sqrtf((float)head_dim);

    for (int layer = 0; layer < cfg->dec_layers; layer++) {
        qwen_dec_layer_t *l = &dec->layers[layer];

        /* Input RMSNorm */
        qwen_rms_norm(x_norm, x, l->input_norm, seq_len, dim, eps);

        /* QKV projections (no bias) */
        qwen_linear_nobias_bf16(q, x_norm, l->wq_weight_bf16, seq_len, dim, q_dim);
        qwen_linear_nobias_bf16(k, x_norm, l->wk_weight_bf16, seq_len, dim, kv_dim);
        qwen_linear_nobias_bf16(v, x_norm, l->wv_weight_bf16, seq_len, dim, kv_dim);

        /* Per-head Q/K RMSNorm */
        qwen_rms_norm_per_head(q, l->q_norm_weight, seq_len, n_heads, head_dim, eps);
        qwen_rms_norm_per_head(k, l->k_norm_weight, seq_len, n_kv_heads, head_dim, eps);

        /* Apply NeoX RoPE */
        qwen_apply_rope_neox(q, rope_cos, rope_sin, seq_len, n_heads, head_dim);
        qwen_apply_rope_neox(k, rope_cos, rope_sin, seq_len, n_kv_heads, head_dim);

        /* Store K, V in cache */
        for (int s = 0; s < seq_len; s++) {
            memcpy(kv_cache_k_at(ctx, layer, start_pos + s),
                   k + s * kv_dim, kv_dim * sizeof(float));
            memcpy(kv_cache_v_at(ctx, layer, start_pos + s),
                   v + s * kv_dim, kv_dim * sizeof(float));
        }

        /* Causal attention */
        int total_seq = start_pos + seq_len;
        float *full_k = kv_cache_k_at(ctx, layer, 0);
        float *full_v = kv_cache_v_at(ctx, layer, 0);
        qwen_causal_attention(attn_out, q, full_k, full_v,
                               seq_len, total_seq, n_heads, n_kv_heads,
                               head_dim, scale, start_pos);

        /* Output projection + residual */
        qwen_linear_nobias_bf16(proj_out, attn_out, l->wo_weight_bf16,
                                 seq_len, q_dim, dim);
        qwen_add_inplace(x, proj_out, seq_len * dim);

        /* Post-attention RMSNorm */
        qwen_rms_norm(x_norm, x, l->post_attn_norm, seq_len, dim, eps);

        if (cfg->is_moe) {
            /* MoE: process each token position independently (each routes
             * to different experts). Adequate for ASR prompt lengths. */
            for (int s = 0; s < seq_len; s++) {
                float *xs = x + (size_t)s * dim;
                const float *xn = x_norm + (size_t)s * dim;
                moe_forward_single(ctx, xs, xn, l);
            }
        } else {
            /* Dense SwiGLU MLP */
            qwen_linear_nobias_bf16(gate_up, x_norm, l->gate_up_fused_bf16,
                                     seq_len, dim, 2 * intermediate);
            qwen_swiglu_multiply(gate, gate_up, seq_len, intermediate);
            qwen_linear_nobias_bf16(ffn_out, gate, l->down_weight_bf16,
                                     seq_len, intermediate, dim);
            qwen_add_inplace(x, ffn_out, seq_len * dim);
        }

    }

    ctx->kv_cache_len = start_pos + seq_len;
}

/* ========================================================================
 * Decoder Forward (Single Token Generation)
 * ======================================================================== */

static void ensure_dec_buffers(qwen_ctx_t *ctx) {
    if (ctx->dec_x) return;
    const qwen_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int q_dim = cfg->dec_heads * cfg->dec_head_dim;
    int kv_dim = cfg->dec_kv_heads * cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;
    int head_dim = cfg->dec_head_dim;

    ctx->dec_x        = (float *)malloc(dim * sizeof(float));
    ctx->dec_x_norm   = (float *)malloc(dim * sizeof(float));
    ctx->dec_q        = (float *)malloc(q_dim * sizeof(float));
    ctx->dec_k        = (float *)malloc(kv_dim * sizeof(float));
    ctx->dec_v        = (float *)malloc(kv_dim * sizeof(float));
    ctx->dec_attn_out = (float *)malloc(q_dim * sizeof(float));
    ctx->dec_proj_out = (float *)malloc(dim * sizeof(float));
    ctx->dec_gate     = (float *)malloc(2 * intermediate * sizeof(float));
    ctx->dec_up       = NULL; /* unused: gate buffer holds fused gate+up */
    ctx->dec_ffn_out  = (float *)malloc(dim * sizeof(float));
    ctx->dec_rope_cos = (float *)malloc(head_dim * sizeof(float));
    ctx->dec_rope_sin = (float *)malloc(head_dim * sizeof(float));
}

int qwen_decoder_forward(qwen_ctx_t *ctx, const float *input_embed) {
    qwen_decoder_t *dec = &ctx->decoder;
    const qwen_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int n_heads = cfg->dec_heads;
    int n_kv_heads = cfg->dec_kv_heads;
    int head_dim = cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;
    float eps = cfg->dec_rms_norm_eps;
    float theta = cfg->dec_rope_theta;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    ensure_dec_buffers(ctx);
    float *x = ctx->dec_x;
    float *x_norm = ctx->dec_x_norm;
    float *q = ctx->dec_q;
    float *k = ctx->dec_k;
    float *v = ctx->dec_v;
    float *attn_out = ctx->dec_attn_out;
    float *proj_out = ctx->dec_proj_out;
    float *gate_buf = ctx->dec_gate;
    float *ffn_out = ctx->dec_ffn_out;
    memcpy(x, input_embed, dim * sizeof(float));

    int pos = ctx->kv_cache_len;

    /* Grow KV cache if needed */
    if (pos >= ctx->kv_cache_max) {
        if (kv_cache_grow(ctx, pos + 1024) != 0) return QWEN_TOKEN_IM_END;
    }

    if (ensure_rope_cache(ctx, pos + 1, head_dim, theta) != 0) {
        return QWEN_TOKEN_IM_END;
    }
    const float *rope_cos = ctx->rope_cache_cos + (size_t)pos * head_dim;
    const float *rope_sin = ctx->rope_cache_sin + (size_t)pos * head_dim;

    float scale = 1.0f / sqrtf((float)head_dim);

    for (int layer = 0; layer < cfg->dec_layers; layer++) {
        qwen_dec_layer_t *l = &dec->layers[layer];

        qwen_rms_norm(x_norm, x, l->input_norm, 1, dim, eps);
        qwen_linear_nobias_bf16_qkv(q, k, v, x_norm,
                                    l->wq_weight_bf16,
                                    l->wk_weight_bf16,
                                    l->wv_weight_bf16,
                                    dim, q_dim, kv_dim);

        /* Per-head Q/K RMSNorm */
        qwen_rms_norm_per_head(q, l->q_norm_weight, 1, n_heads, head_dim, eps);
        qwen_rms_norm_per_head(k, l->k_norm_weight, 1, n_kv_heads, head_dim, eps);

        /* Apply NeoX RoPE */
        qwen_apply_rope_neox(q, rope_cos, rope_sin, 1, n_heads, head_dim);
        qwen_apply_rope_neox(k, rope_cos, rope_sin, 1, n_kv_heads, head_dim);

        memcpy(kv_cache_k_at(ctx, layer, pos), k, kv_dim * sizeof(float));
        memcpy(kv_cache_v_at(ctx, layer, pos), v, kv_dim * sizeof(float));

        int total_seq = pos + 1;
        float *full_k = kv_cache_k_at(ctx, layer, 0);
        float *full_v = kv_cache_v_at(ctx, layer, 0);

        qwen_causal_attention(attn_out, q, full_k, full_v,
                               1, total_seq, n_heads, n_kv_heads,
                               head_dim, scale, pos);

        qwen_linear_nobias_bf16(proj_out, attn_out, l->wo_weight_bf16, 1, q_dim, dim);
        qwen_add_inplace(x, proj_out, dim);

        qwen_rms_norm(x_norm, x, l->post_attn_norm, 1, dim, eps);

        if (cfg->is_moe) {
            /* MoE: route to top-k experts */
            moe_forward_single(ctx, x, x_norm, l);
        } else {
            /* Fused gate+up matvec: one pass over x_norm, output interleaved [g0,u0,g1,u1,...] */
            qwen_linear_nobias_bf16(gate_buf, x_norm, l->gate_up_fused_bf16,
                                     1, dim, 2 * intermediate);
            /* In-place for seq=1: gate_buf[0:inter] receives SwiGLU output. */
            qwen_swiglu_multiply(gate_buf, gate_buf, 1, intermediate);
            qwen_linear_nobias_bf16(ffn_out, gate_buf, l->down_weight_bf16, 1, intermediate, dim);
            qwen_add_inplace(x, ffn_out, dim);
        }
    }

    ctx->kv_cache_len = pos + 1;

    /* Final norm + streaming argmax (no logits buffer needed) */
    qwen_rms_norm(x, x, dec->norm, 1, dim, eps);
    return qwen_argmax_matvec_bf16(x, dec->tok_embeddings_bf16, dim, cfg->vocab_size);
}

/* ========================================================================
 * Decoder Forward — Full Logits Variant (for sampling paths)
 * ======================================================================== */

void qwen_decoder_forward_logits(qwen_ctx_t *ctx, const float *input_embed, float *logits) {
    qwen_decoder_t *dec = &ctx->decoder;
    const qwen_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int n_heads = cfg->dec_heads;
    int n_kv_heads = cfg->dec_kv_heads;
    int head_dim = cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;
    float eps = cfg->dec_rms_norm_eps;
    float theta = cfg->dec_rope_theta;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    ensure_dec_buffers(ctx);
    float *x = ctx->dec_x;
    float *x_norm = ctx->dec_x_norm;
    float *q = ctx->dec_q;
    float *k = ctx->dec_k;
    float *v = ctx->dec_v;
    float *attn_out = ctx->dec_attn_out;
    float *proj_out = ctx->dec_proj_out;
    float *gate_buf = ctx->dec_gate;
    float *ffn_out = ctx->dec_ffn_out;
    memcpy(x, input_embed, dim * sizeof(float));

    int pos = ctx->kv_cache_len;

    if (pos >= ctx->kv_cache_max) {
        if (kv_cache_grow(ctx, pos + 1024) != 0) {
            /* On failure, produce zero logits */
            memset(logits, 0, (size_t)cfg->vocab_size * sizeof(float));
            return;
        }
    }

    if (ensure_rope_cache(ctx, pos + 1, head_dim, theta) != 0) {
        memset(logits, 0, (size_t)cfg->vocab_size * sizeof(float));
        return;
    }
    const float *rope_cos = ctx->rope_cache_cos + (size_t)pos * head_dim;
    const float *rope_sin = ctx->rope_cache_sin + (size_t)pos * head_dim;

    float scale = 1.0f / sqrtf((float)head_dim);

    for (int layer = 0; layer < cfg->dec_layers; layer++) {
        qwen_dec_layer_t *l = &dec->layers[layer];

        qwen_rms_norm(x_norm, x, l->input_norm, 1, dim, eps);
        qwen_linear_nobias_bf16_qkv(q, k, v, x_norm,
                                    l->wq_weight_bf16,
                                    l->wk_weight_bf16,
                                    l->wv_weight_bf16,
                                    dim, q_dim, kv_dim);

        qwen_rms_norm_per_head(q, l->q_norm_weight, 1, n_heads, head_dim, eps);
        qwen_rms_norm_per_head(k, l->k_norm_weight, 1, n_kv_heads, head_dim, eps);

        qwen_apply_rope_neox(q, rope_cos, rope_sin, 1, n_heads, head_dim);
        qwen_apply_rope_neox(k, rope_cos, rope_sin, 1, n_kv_heads, head_dim);

        memcpy(kv_cache_k_at(ctx, layer, pos), k, kv_dim * sizeof(float));
        memcpy(kv_cache_v_at(ctx, layer, pos), v, kv_dim * sizeof(float));

        int total_seq = pos + 1;
        float *full_k = kv_cache_k_at(ctx, layer, 0);
        float *full_v = kv_cache_v_at(ctx, layer, 0);

        qwen_causal_attention(attn_out, q, full_k, full_v,
                               1, total_seq, n_heads, n_kv_heads,
                               head_dim, scale, pos);

        qwen_linear_nobias_bf16(proj_out, attn_out, l->wo_weight_bf16, 1, q_dim, dim);
        qwen_add_inplace(x, proj_out, dim);

        qwen_rms_norm(x_norm, x, l->post_attn_norm, 1, dim, eps);

        if (cfg->is_moe) {
            moe_forward_single(ctx, x, x_norm, l);
        } else {
            qwen_linear_nobias_bf16(gate_buf, x_norm, l->gate_up_fused_bf16,
                                     1, dim, 2 * intermediate);
            qwen_swiglu_multiply(gate_buf, gate_buf, 1, intermediate);
            qwen_linear_nobias_bf16(ffn_out, gate_buf, l->down_weight_bf16, 1, intermediate, dim);
            qwen_add_inplace(x, ffn_out, dim);
        }
    }

    ctx->kv_cache_len = pos + 1;

    /* Final norm + full lm_head matmul → logits */
    qwen_rms_norm(x, x, dec->norm, 1, dim, eps);
    qwen_matmul_t_bf16(logits, x, dec->tok_embeddings_bf16, 1, dim, cfg->vocab_size);
}
