/*
 * qwen25_omni_encoder.c - Qwen2.5-Omni-7B audio encoder
 *
 * Architecture:
 *   Conv1D stem: conv1 [128→1280, k=3, s=1, pad=1] → GELU
 *                conv2 [1280→1280, k=3, s=2, pad=1] → GELU
 *   Transpose [1280, T/2] → [T/2, 1280]
 *   Sinusoidal position embeddings
 *   32 transformer layers (windowed attention):
 *     LayerNorm → MHA (Q/V with bias, K without, O with bias) → residual
 *     LayerNorm → GELU FFN → residual
 *   Final LayerNorm
 *   Single projection: proj [1280→3584]
 *   Prepend audio_bos, append audio_eos embeddings
 */

#include "qwen25_omni.h"
#include "qwen_asr_kernels.h"
#include "qwen_asr_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================
 * Weight Loading
 * ======================================================================== */

#define ENC_PREFIX "thinker.audio_tower."

static float *load_f32(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "q25 encoder: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_f32(sf, t);
}

static float *load_bf16_as_f32(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "q25 encoder: weight not found: %s\n", name);
        return NULL;
    }
    uint16_t *bf16 = safetensors_get_bf16_direct(sf, t);
    if (!bf16) return NULL;

    size_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];

    float *f32 = (float *)malloc(n * sizeof(float));
    if (!f32) return NULL;

    uint32_t *d = (uint32_t *)(void *)f32;
    for (size_t i = 0; i < n; i++)
        d[i] = ((uint32_t)bf16[i]) << 16;

    return f32;
}

/* Try bf16→f32 first, fall back to f32 if tensor is already f32 */
static float *load_weight_as_f32(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "q25 encoder: weight not found: %s\n", name);
        return NULL;
    }
    if (safetensor_is_bf16(t)) {
        return load_bf16_as_f32(ms, name);
    }
    return safetensors_get_f32(sf, t);
}

int q25_encoder_load(q25_encoder_t *enc, void *ms_void, const q25_config_t *cfg) {
    multi_safetensors_t *ms = (multi_safetensors_t *)ms_void;
    char name[512];

    /* Conv1D stem */
    snprintf(name, sizeof(name), "%sconv1.weight", ENC_PREFIX);
    enc->conv1_weight = load_weight_as_f32(ms, name);
    snprintf(name, sizeof(name), "%sconv1.bias", ENC_PREFIX);
    enc->conv1_bias = load_f32(ms, name);
    snprintf(name, sizeof(name), "%sconv2.weight", ENC_PREFIX);
    enc->conv2_weight = load_weight_as_f32(ms, name);
    snprintf(name, sizeof(name), "%sconv2.bias", ENC_PREFIX);
    enc->conv2_bias = load_f32(ms, name);

    if (!enc->conv1_weight || !enc->conv2_weight) return -1;

    /* Transformer layers */
    for (int i = 0; i < cfg->enc_layers; i++) {
        q25_enc_layer_t *l = &enc->layers[i];
        const char *lp = ENC_PREFIX "layers";

        /* Attention: Q has bias, K has NO bias, V has bias, O has bias */
        snprintf(name, sizeof(name), "%s.%d.self_attn.q_proj.weight", lp, i);
        l->wq_weight = load_bf16_as_f32(ms, name);
        snprintf(name, sizeof(name), "%s.%d.self_attn.q_proj.bias", lp, i);
        l->wq_bias = load_f32(ms, name);

        snprintf(name, sizeof(name), "%s.%d.self_attn.k_proj.weight", lp, i);
        l->wk_weight = load_bf16_as_f32(ms, name);
        /* K has no bias in Qwen2.5-Omni encoder */

        snprintf(name, sizeof(name), "%s.%d.self_attn.v_proj.weight", lp, i);
        l->wv_weight = load_bf16_as_f32(ms, name);
        snprintf(name, sizeof(name), "%s.%d.self_attn.v_proj.bias", lp, i);
        l->wv_bias = load_f32(ms, name);

        snprintf(name, sizeof(name), "%s.%d.self_attn.out_proj.weight", lp, i);
        l->wo_weight = load_bf16_as_f32(ms, name);
        snprintf(name, sizeof(name), "%s.%d.self_attn.out_proj.bias", lp, i);
        l->wo_bias = load_f32(ms, name);

        /* Pre-attention LayerNorm */
        snprintf(name, sizeof(name), "%s.%d.self_attn_layer_norm.weight", lp, i);
        l->attn_norm_weight = load_f32(ms, name);
        snprintf(name, sizeof(name), "%s.%d.self_attn_layer_norm.bias", lp, i);
        l->attn_norm_bias = load_f32(ms, name);

        /* FFN */
        snprintf(name, sizeof(name), "%s.%d.fc1.weight", lp, i);
        l->fc1_weight = load_bf16_as_f32(ms, name);
        snprintf(name, sizeof(name), "%s.%d.fc1.bias", lp, i);
        l->fc1_bias = load_f32(ms, name);
        snprintf(name, sizeof(name), "%s.%d.fc2.weight", lp, i);
        l->fc2_weight = load_bf16_as_f32(ms, name);
        snprintf(name, sizeof(name), "%s.%d.fc2.bias", lp, i);
        l->fc2_bias = load_f32(ms, name);

        /* Pre-FFN LayerNorm */
        snprintf(name, sizeof(name), "%s.%d.final_layer_norm.weight", lp, i);
        l->ffn_norm_weight = load_f32(ms, name);
        snprintf(name, sizeof(name), "%s.%d.final_layer_norm.bias", lp, i);
        l->ffn_norm_bias = load_f32(ms, name);

        if (!l->wq_weight || !l->wk_weight ||
            !l->wv_weight || !l->wo_weight) {
            fprintf(stderr, "q25 encoder: failed to load layer %d weights\n", i);
            return -1;
        }
    }

    /* Final LayerNorm */
    snprintf(name, sizeof(name), "%sln_post.weight", ENC_PREFIX);
    enc->ln_post_weight = load_f32(ms, name);
    snprintf(name, sizeof(name), "%sln_post.bias", ENC_PREFIX);
    enc->ln_post_bias = load_f32(ms, name);

    /* Single projection (not proj1+proj2 like Qwen3) */
    snprintf(name, sizeof(name), "%sproj.weight", ENC_PREFIX);
    enc->proj_weight = load_bf16_as_f32(ms, name);
    snprintf(name, sizeof(name), "%sproj.bias", ENC_PREFIX);
    enc->proj_bias = load_f32(ms, name);

    /* Learned audio BOS/EOS boundary tokens [2, output_dim] */
    snprintf(name, sizeof(name), "%saudio_bos_eos_token.weight", ENC_PREFIX);
    enc->audio_bos_eos = load_weight_as_f32(ms, name);

    if (!enc->ln_post_weight || !enc->proj_weight || !enc->audio_bos_eos)
        return -1;

    return 0;
}

/* ========================================================================
 * Forward Pass
 * ======================================================================== */

float *q25_encoder_forward(q25_ctx_t *ctx, const float *mel, int mel_frames,
                            int *out_seq_len) {
    const q25_config_t *cfg = &ctx->config;
    q25_encoder_t *enc = &ctx->encoder;

    int d_model = cfg->enc_d_model;
    int n_heads = cfg->enc_heads;
    int head_dim = cfg->enc_head_dim;
    int ffn_dim = cfg->enc_ffn_dim;
    int output_dim = cfg->enc_output_dim;
    int n_window = cfg->enc_n_window;

    /* ---- Conv1D stem ----
     * mel: [128, mel_frames]
     * conv1: [128→1280, k=3, s=1, pad=1] → [1280, mel_frames]
     * conv2: [1280→1280, k=3, s=2, pad=1] → [1280, T/2]
     */
    int l_after_c1 = (mel_frames + 2 * 1 - 3) / 1 + 1; /* same as mel_frames */
    float *c1 = (float *)malloc((size_t)d_model * l_after_c1 * sizeof(float));
    qwen_conv1d(c1, mel, enc->conv1_weight, enc->conv1_bias,
                128, d_model, mel_frames, 3, 1, 1);
    qwen_gelu(c1, d_model * l_after_c1);

    int l_after_c2 = (l_after_c1 + 2 * 1 - 3) / 2 + 1;
    float *c2 = (float *)malloc((size_t)d_model * l_after_c2 * sizeof(float));
    qwen_conv1d(c2, c1, enc->conv2_weight, enc->conv2_bias,
                d_model, d_model, l_after_c1, 3, 2, 1);
    qwen_gelu(c2, d_model * l_after_c2);
    free(c1);

    int total_tokens = l_after_c2;

    /* Transpose [d_model, total_tokens] → [total_tokens, d_model] */
    float *x = (float *)calloc((size_t)total_tokens * d_model, sizeof(float));
    for (int t = 0; t < total_tokens; t++) {
        for (int d = 0; d < d_model; d++) {
            x[t * d_model + d] = c2[d * total_tokens + t];
        }
    }
    free(c2);

    /* Add sinusoidal position embeddings */
    float *pe = (float *)malloc((size_t)total_tokens * d_model * sizeof(float));
    qwen_sinusoidal_pe(pe, total_tokens, d_model);
    qwen_add_inplace(x, pe, total_tokens * d_model);
    free(pe);

    /* ---- Build attention window boundaries ---- */
    int n_windows = (total_tokens + n_window - 1) / n_window;
    int *window_starts = (int *)malloc((n_windows + 1) * sizeof(int));
    for (int w = 0; w < n_windows; w++) {
        window_starts[w] = w * n_window;
    }
    window_starts[n_windows] = total_tokens;

    /* ---- Transformer layers ---- */
    float *x_norm = (float *)malloc((size_t)total_tokens * d_model * sizeof(float));
    float *q = (float *)malloc((size_t)total_tokens * d_model * sizeof(float));
    float *k = (float *)malloc((size_t)total_tokens * d_model * sizeof(float));
    float *v = (float *)malloc((size_t)total_tokens * d_model * sizeof(float));
    float *attn_out = (float *)malloc((size_t)total_tokens * d_model * sizeof(float));
    float *proj_out = (float *)malloc((size_t)total_tokens * d_model * sizeof(float));
    float *ffn_mid = (float *)malloc((size_t)total_tokens * ffn_dim * sizeof(float));
    float *ffn_out = (float *)malloc((size_t)total_tokens * d_model * sizeof(float));

    float scale = 1.0f / sqrtf((float)head_dim);

    for (int layer = 0; layer < cfg->enc_layers; layer++) {
        q25_enc_layer_t *l = &enc->layers[layer];

        /* ---- Self-attention ---- */
        qwen_layer_norm(x_norm, x, l->attn_norm_weight, l->attn_norm_bias,
                        total_tokens, d_model, 1e-5f);

        /* Q with bias */
        qwen_linear(q, x_norm, l->wq_weight, l->wq_bias,
                     total_tokens, d_model, d_model);
        /* K without bias */
        qwen_linear_nobias(k, x_norm, l->wk_weight,
                            total_tokens, d_model, d_model);
        /* V with bias */
        qwen_linear(v, x_norm, l->wv_weight, l->wv_bias,
                     total_tokens, d_model, d_model);

        qwen_bidirectional_attention(attn_out, q, k, v,
                                      total_tokens, n_heads, head_dim, scale,
                                      window_starts, n_windows);

        /* Output projection with bias + residual */
        qwen_linear(proj_out, attn_out, l->wo_weight, l->wo_bias,
                     total_tokens, d_model, d_model);
        qwen_add_inplace(x, proj_out, total_tokens * d_model);

        /* ---- FFN ---- */
        qwen_layer_norm(x_norm, x, l->ffn_norm_weight, l->ffn_norm_bias,
                        total_tokens, d_model, 1e-5f);

        qwen_linear(ffn_mid, x_norm, l->fc1_weight, l->fc1_bias,
                     total_tokens, d_model, ffn_dim);
        qwen_gelu(ffn_mid, total_tokens * ffn_dim);
        qwen_linear(ffn_out, ffn_mid, l->fc2_weight, l->fc2_bias,
                     total_tokens, ffn_dim, d_model);
        qwen_add_inplace(x, ffn_out, total_tokens * d_model);
    }

    /* Final LayerNorm */
    qwen_layer_norm(x, x, enc->ln_post_weight, enc->ln_post_bias,
                    total_tokens, d_model, 1e-5f);

    /* Single projection: [total_tokens, 1280] → [total_tokens, 3584] */
    float *projected = (float *)malloc((size_t)total_tokens * output_dim * sizeof(float));
    qwen_linear(projected, x, enc->proj_weight, enc->proj_bias,
                 total_tokens, d_model, output_dim);

    /* Prepend audio_bos, append audio_eos → [total_tokens+2, output_dim] */
    int final_len = total_tokens + 2;
    float *enc_output = (float *)malloc((size_t)final_len * output_dim * sizeof(float));

    /* Row 0: audio_bos (first row of audio_bos_eos) */
    memcpy(enc_output, enc->audio_bos_eos, output_dim * sizeof(float));
    /* Rows 1..total_tokens: projected encoder output */
    memcpy(enc_output + output_dim, projected, (size_t)total_tokens * output_dim * sizeof(float));
    /* Last row: audio_eos (second row of audio_bos_eos) */
    memcpy(enc_output + (size_t)(total_tokens + 1) * output_dim,
           enc->audio_bos_eos + output_dim, output_dim * sizeof(float));

    free(projected);

    /* Clean up */
    free(x); free(x_norm); free(q); free(k); free(v);
    free(attn_out); free(proj_out);
    free(ffn_mid); free(ffn_out);
    free(window_starts);

    *out_seq_len = final_len;
    return enc_output;
}
