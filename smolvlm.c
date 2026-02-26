/*
 * smolvlm.c - SmolVLM-Instruct model loading and generation orchestration
 *
 * Loads vision encoder, connector, decoder, and tokenizer from a model directory.
 * Provides smolvlm_generate() for end-to-end image+text generation.
 */

#include "smolvlm.h"
#include "smolvlm_tokenizer.h"
#include "qwen_asr_kernels.h"
#include "qwen_asr_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Global verbose flag (defined here since we don't link qwen_asr.c) */
int qwen_verbose = 0;

/* Forward declarations for loading functions in other TUs */
extern int smolvlm_vision_load(smolvlm_vision_t *vis, smolvlm_connector_t *conn,
                                multi_safetensors_t *ms, const smolvlm_config_t *cfg);
extern int smolvlm_decoder_load(smolvlm_decoder_t *dec, multi_safetensors_t *ms,
                                 const smolvlm_config_t *cfg);

/* ========================================================================
 * Config Loading (from config.json)
 * ======================================================================== */

/* Minimal JSON value extraction helpers */
static const char *find_key(const char *json, const char *key) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return NULL;
    p += strlen(pattern);
    while (*p == ' ' || *p == ':' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

static int json_int(const char *json, const char *key, int def) {
    const char *p = find_key(json, key);
    if (!p) return def;
    int neg = 0;
    if (*p == '-') { neg = 1; p++; }
    int val = 0;
    while (*p >= '0' && *p <= '9') {
        val = val * 10 + (*p - '0');
        p++;
    }
    return neg ? -val : val;
}

static double json_float(const char *json, const char *key, double def) {
    const char *p = find_key(json, key);
    if (!p) return def;
    char buf[64];
    int i = 0;
    while (i < 63 && ((*p >= '0' && *p <= '9') || *p == '.' || *p == '-' || *p == 'e' || *p == 'E' || *p == '+'))
        buf[i++] = *p++;
    buf[i] = '\0';
    return atof(buf);
}

/* Find a sub-object and return pointer to its content (after '{') */
static const char *find_object(const char *json, const char *key) {
    const char *p = find_key(json, key);
    if (!p || *p != '{') return NULL;
    return p + 1;
}

static int load_config(smolvlm_config_t *cfg, const char *model_dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", model_dir);

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "smolvlm: cannot open %s\n", path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *json = (char *)malloc((size_t)size + 1);
    if (!json || fread(json, 1, (size_t)size, f) != (size_t)size) {
        fclose(f);
        free(json);
        return -1;
    }
    fclose(f);
    json[size] = '\0';

    /* Top-level config */
    cfg->scale_factor = json_int(json, "scale_factor", 3);
    cfg->image_seq_len = json_int(json, "image_seq_len", 81);

    /* Vision config */
    const char *vc = find_object(json, "vision_config");
    if (vc) {
        cfg->vis_hidden = json_int(vc, "hidden_size", 1152);
        cfg->vis_heads = json_int(vc, "num_attention_heads", 16);
        cfg->vis_layers = json_int(vc, "num_hidden_layers", 27);
        cfg->vis_ffn_dim = json_int(vc, "intermediate_size", 4304);
        cfg->vis_image_size = json_int(vc, "image_size", 384);
        cfg->vis_patch_size = json_int(vc, "patch_size", 14);
        cfg->vis_layer_norm_eps = (float)json_float(vc, "layer_norm_eps", 1e-6);
    } else {
        /* Defaults */
        cfg->vis_hidden = 1152;
        cfg->vis_heads = 16;
        cfg->vis_layers = 27;
        cfg->vis_ffn_dim = 4304;
        cfg->vis_image_size = 384;
        cfg->vis_patch_size = 14;
        cfg->vis_layer_norm_eps = 1e-6f;
    }
    cfg->vis_head_dim = cfg->vis_hidden / cfg->vis_heads;

    /* Text config */
    const char *tc = find_object(json, "text_config");
    if (tc) {
        cfg->dec_hidden = json_int(tc, "hidden_size", 2048);
        cfg->dec_heads = json_int(tc, "num_attention_heads", 32);
        cfg->dec_kv_heads = json_int(tc, "num_key_value_heads", 32);
        cfg->dec_layers = json_int(tc, "num_hidden_layers", 24);
        cfg->dec_intermediate = json_int(tc, "intermediate_size", 8192);
        cfg->vocab_size = json_int(tc, "vocab_size", 49155);
        cfg->dec_rms_norm_eps = (float)json_float(tc, "rms_norm_eps", 1e-5);
        cfg->dec_rope_theta = (float)json_float(tc, "rope_theta", 273768.0);
        int hd = json_int(tc, "head_dim", 0);
        cfg->dec_head_dim = hd > 0 ? hd : cfg->dec_hidden / cfg->dec_heads;
    } else {
        cfg->dec_hidden = 2048;
        cfg->dec_heads = 32;
        cfg->dec_kv_heads = 32;
        cfg->dec_layers = 24;
        cfg->dec_intermediate = 8192;
        cfg->vocab_size = 49155;
        cfg->dec_rms_norm_eps = 1e-5f;
        cfg->dec_rope_theta = 273768.0f;
        cfg->dec_head_dim = 64;
    }

    free(json);
    return 0;
}

/* ========================================================================
 * Model Loading
 * ======================================================================== */

smolvlm_ctx_t *smolvlm_load(const char *model_dir) {
    smolvlm_ctx_t *ctx = (smolvlm_ctx_t *)calloc(1, sizeof(smolvlm_ctx_t));
    if (!ctx) return NULL;

    snprintf(ctx->model_dir, sizeof(ctx->model_dir), "%s", model_dir);

    /* Load config */
    if (load_config(&ctx->config, model_dir) != 0) {
        free(ctx);
        return NULL;
    }

    if (qwen_verbose >= 1) {
        fprintf(stderr, "SmolVLM: vision %d layers (hidden=%d), decoder %d layers (hidden=%d)\n",
                ctx->config.vis_layers, ctx->config.vis_hidden,
                ctx->config.dec_layers, ctx->config.dec_hidden);
        fprintf(stderr, "  image %dx%d patch=%d, scale=%d, seq_len=%d\n",
                ctx->config.vis_image_size, ctx->config.vis_image_size,
                ctx->config.vis_patch_size, ctx->config.scale_factor,
                ctx->config.image_seq_len);
        fprintf(stderr, "  decoder: heads=%d kv_heads=%d head_dim=%d rope_theta=%.0f\n",
                ctx->config.dec_heads, ctx->config.dec_kv_heads,
                ctx->config.dec_head_dim, ctx->config.dec_rope_theta);
    }

    /* Open safetensors */
    ctx->safetensors = multi_safetensors_open(model_dir);
    if (!ctx->safetensors) {
        fprintf(stderr, "smolvlm: failed to open safetensors in %s\n", model_dir);
        free(ctx);
        return NULL;
    }

    /* Load weights */
    if (qwen_verbose >= 1)
        fprintf(stderr, "  Loading vision encoder + connector...\n");
    if (smolvlm_vision_load(&ctx->vision, &ctx->connector,
                             (multi_safetensors_t *)ctx->safetensors,
                             &ctx->config) != 0) {
        fprintf(stderr, "smolvlm: failed to load vision weights\n");
        smolvlm_free(ctx);
        return NULL;
    }

    if (qwen_verbose >= 1)
        fprintf(stderr, "  Loading decoder...\n");
    if (smolvlm_decoder_load(&ctx->decoder,
                              (multi_safetensors_t *)ctx->safetensors,
                              &ctx->config) != 0) {
        fprintf(stderr, "smolvlm: failed to load decoder weights\n");
        smolvlm_free(ctx);
        return NULL;
    }

    /* Load tokenizer */
    if (qwen_verbose >= 1)
        fprintf(stderr, "  Loading tokenizer...\n");
    ctx->tokenizer = smolvlm_tokenizer_load(model_dir);
    if (!ctx->tokenizer) {
        fprintf(stderr, "smolvlm: failed to load tokenizer\n");
        smolvlm_free(ctx);
        return NULL;
    }

    if (qwen_verbose >= 1)
        fprintf(stderr, "  Model loaded.\n");

    return ctx;
}

/* ========================================================================
 * Cleanup
 * ======================================================================== */

void smolvlm_free(smolvlm_ctx_t *ctx) {
    if (!ctx) return;

    /* Vision encoder layer weights (allocated by load_auto_f32) */
    smolvlm_vision_t *vis = &ctx->vision;
    free(vis->patch_weight);
    free(vis->patch_bias);
    free(vis->position_embedding);
    for (int i = 0; i < ctx->config.vis_layers; i++) {
        smolvlm_vis_layer_t *l = &vis->layers[i];
        free(l->ln1_weight); free(l->ln1_bias);
        free(l->wq_weight); free(l->wq_bias);
        free(l->wk_weight); free(l->wk_bias);
        free(l->wv_weight); free(l->wv_bias);
        free(l->wo_weight); free(l->wo_bias);
        free(l->ln2_weight); free(l->ln2_bias);
        free(l->fc1_weight); free(l->fc1_bias);
        free(l->fc2_weight); free(l->fc2_bias);
    }
    free(vis->post_ln_weight);
    free(vis->post_ln_bias);

    /* Connector */
    free(ctx->connector.proj_weight);

    /* Decoder fused weights */
    for (int i = 0; i < ctx->config.dec_layers; i++) {
        free(ctx->decoder.layers[i].gate_up_fused_bf16);
        /* Note: other bf16 weights are mmap'd, not freed */
        free(ctx->decoder.layers[i].input_norm);
        free(ctx->decoder.layers[i].post_attn_norm);
    }
    free(ctx->decoder.norm);

    /* Safetensors */
    if (ctx->safetensors)
        multi_safetensors_close((multi_safetensors_t *)ctx->safetensors);

    /* Tokenizer */
    smolvlm_tokenizer_free(ctx->tokenizer);

    /* Buffers */
    free(ctx->kv_cache_k);
    free(ctx->kv_cache_v);
    free(ctx->dec_x);
    free(ctx->dec_x_norm);
    free(ctx->dec_q);
    free(ctx->dec_k);
    free(ctx->dec_v);
    free(ctx->dec_attn_out);
    free(ctx->dec_proj_out);
    free(ctx->dec_gate);
    free(ctx->dec_ffn_out);
    free(ctx->dec_rope_cos);
    free(ctx->dec_rope_sin);
    free(ctx->pref_x);
    free(ctx->pref_x_norm);
    free(ctx->pref_q);
    free(ctx->pref_k);
    free(ctx->pref_v);
    free(ctx->pref_attn_out);
    free(ctx->pref_proj_out);
    free(ctx->pref_ffn_out);
    free(ctx->pref_gate);
    free(ctx->pref_gate_up);
    free(ctx->rope_cache_cos);
    free(ctx->rope_cache_sin);
    free(ctx->rope_inv_freq);

    free(ctx);
}

void smolvlm_set_token_callback(smolvlm_ctx_t *ctx, smolvlm_token_cb cb, void *userdata) {
    ctx->token_cb = cb;
    ctx->token_cb_userdata = userdata;
}

/* ========================================================================
 * Token Embedding Helper
 * ======================================================================== */

static void embed_token_bf16(float *out, const uint16_t *embeddings, int token_id,
                              int hidden_dim) {
    const uint16_t *row = embeddings + (size_t)token_id * hidden_dim;
    uint32_t *d = (uint32_t *)(void *)out;
    for (int i = 0; i < hidden_dim; i++)
        d[i] = ((uint32_t)row[i]) << 16;
}

/* ========================================================================
 * Generation
 * ======================================================================== */

static double time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

char *smolvlm_generate(smolvlm_ctx_t *ctx, const char *image_path,
                        const char *prompt, int max_tokens) {
    const smolvlm_config_t *cfg = &ctx->config;
    int hidden = cfg->dec_hidden;

    double t_start = time_ms();

    /* ---- Load and preprocess image ---- */
    int img_w, img_h;
    float *image = smolvlm_load_image(image_path, cfg->vis_image_size, &img_w, &img_h);
    if (!image) {
        fprintf(stderr, "smolvlm: failed to load image %s\n", image_path);
        return NULL;
    }

    /* ---- Vision encoder forward ---- */
    double t_enc_start = time_ms();
    int n_vis_tokens;
    float *vis_embeds = smolvlm_vision_forward(ctx, image, 3, img_h, img_w, &n_vis_tokens);
    free(image);
    if (!vis_embeds) {
        fprintf(stderr, "smolvlm: vision encoder failed\n");
        return NULL;
    }
    double t_enc_end = time_ms();

    if (qwen_verbose >= 1) {
        fprintf(stderr, "  Vision: %d tokens, %.0f ms\n",
                n_vis_tokens, t_enc_end - t_enc_start);
    }

    /* ---- Build prompt token sequence ---- */
    /* Format: <|im_start|>User:<fake_image><image>...<image><fake_image>prompt<eos>\nAssistant: */
    int n_user, n_prompt_text, n_newline, n_assistant;
    int *user_toks = smolvlm_tokenizer_encode(ctx->tokenizer, "User:", &n_user);
    int *prompt_toks = smolvlm_tokenizer_encode(ctx->tokenizer, prompt, &n_prompt_text);
    int *nl_toks = smolvlm_tokenizer_encode(ctx->tokenizer, "\n", &n_newline);
    int *asst_toks = smolvlm_tokenizer_encode(ctx->tokenizer, "Assistant:", &n_assistant);

    int n_image_tokens = cfg->image_seq_len;
    int total_prompt = 1 + n_user + 1 + n_image_tokens + 1 + n_prompt_text + 1 + n_newline + n_assistant;
    int *prompt_ids = (int *)malloc(total_prompt * sizeof(int));
    if (!prompt_ids) {
        free(vis_embeds);
        free(user_toks); free(prompt_toks); free(nl_toks); free(asst_toks);
        return NULL;
    }

    int pos = 0;
    prompt_ids[pos++] = SMOLVLM_TOKEN_IM_START;
    for (int i = 0; i < n_user; i++) prompt_ids[pos++] = user_toks[i];
    prompt_ids[pos++] = SMOLVLM_TOKEN_FAKE_IMAGE;
    for (int i = 0; i < n_image_tokens; i++) prompt_ids[pos++] = SMOLVLM_TOKEN_IMAGE;
    prompt_ids[pos++] = SMOLVLM_TOKEN_FAKE_IMAGE;
    for (int i = 0; i < n_prompt_text; i++) prompt_ids[pos++] = prompt_toks[i];
    prompt_ids[pos++] = SMOLVLM_TOKEN_EOS;
    for (int i = 0; i < n_newline; i++) prompt_ids[pos++] = nl_toks[i];
    for (int i = 0; i < n_assistant; i++) prompt_ids[pos++] = asst_toks[i];

    free(user_toks); free(prompt_toks); free(nl_toks); free(asst_toks);

    if (qwen_verbose >= 1) {
        fprintf(stderr, "  Prompt: %d tokens (%d image + %d text)\n",
                pos, n_image_tokens, pos - n_image_tokens);
    }

    /* ---- Build embeddings: text from tok_embeddings, image from vision output ---- */
    float *embeddings = (float *)malloc((size_t)pos * hidden * sizeof(float));
    if (!embeddings) {
        free(vis_embeds);
        free(prompt_ids);
        return NULL;
    }

    int img_idx = 0;
    for (int i = 0; i < pos; i++) {
        if (prompt_ids[i] == SMOLVLM_TOKEN_IMAGE && img_idx < n_vis_tokens) {
            memcpy(embeddings + (size_t)i * hidden,
                   vis_embeds + (size_t)img_idx * hidden,
                   hidden * sizeof(float));
            img_idx++;
        } else {
            embed_token_bf16(embeddings + (size_t)i * hidden,
                             ctx->decoder.tok_embeddings_bf16,
                             prompt_ids[i], hidden);
        }
    }
    free(vis_embeds);
    free(prompt_ids);

    /* ---- Reset KV cache ---- */
    ctx->kv_cache_len = 0;

    /* ---- Prefill all-but-last, then forward last to get first token ---- */
    double t_dec_start = time_ms();

    if (pos > 1) {
        smolvlm_decoder_prefill(ctx, embeddings, pos - 1);
    }

    /* First generated token from the last prompt embedding */
    int first_token = smolvlm_decoder_forward(ctx,
        embeddings + (size_t)(pos - 1) * hidden);
    free(embeddings);

    /* ---- Autoregressive generation ---- */
    char *result = NULL;
    size_t result_len = 0;
    size_t result_cap = 0;

    int n_generated = 0;
    int token = first_token;

    while (token != SMOLVLM_TOKEN_EOS && n_generated < max_tokens) {
        const char *piece = smolvlm_tokenizer_decode(ctx->tokenizer, token);

        /* Accumulate into result */
        size_t piece_len = strlen(piece);
        if (result_len + piece_len + 1 > result_cap) {
            result_cap = (result_cap == 0) ? 256 : result_cap * 2;
            if (result_len + piece_len + 1 > result_cap)
                result_cap = result_len + piece_len + 1;
            char *tmp = (char *)realloc(result, result_cap);
            if (!tmp) break;
            result = tmp;
        }
        memcpy(result + result_len, piece, piece_len);
        result_len += piece_len;
        result[result_len] = '\0';

        /* Emit via callback */
        if (ctx->token_cb)
            ctx->token_cb(piece, ctx->token_cb_userdata);

        n_generated++;

        /* Next token */
        float tok_embed[SMOLVLM_DEC_HIDDEN];
        embed_token_bf16(tok_embed, ctx->decoder.tok_embeddings_bf16, token, hidden);
        token = smolvlm_decoder_forward(ctx, tok_embed);
    }

    double t_end = time_ms();

    /* Performance stats */
    ctx->perf_total_ms = t_end - t_start;
    ctx->perf_tokens = n_generated;
    ctx->perf_encode_ms = t_enc_end - t_enc_start;
    ctx->perf_decode_ms = t_end - t_dec_start;

    return result;
}
