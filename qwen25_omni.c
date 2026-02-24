/*
 * qwen25_omni.c - Qwen2.5-Omni-7B load, orchestration, prompt building
 *
 * Separate from Qwen3 pipeline. Uses q25_* types and API.
 */

#include "qwen25_omni.h"
#include "qwen_asr_kernels.h"
#include "qwen_asr_safetensors.h"
#include "qwen_asr_audio.h"
#include "qwen_asr_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

extern int qwen_verbose;

static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* ========================================================================
 * Token Callback / Prompt
 * ======================================================================== */

void q25_set_token_callback(q25_ctx_t *ctx, qwen_token_cb cb, void *ud) {
    ctx->token_cb = cb;
    ctx->token_cb_userdata = ud;
}

int q25_set_prompt(q25_ctx_t *ctx, const char *prompt) {
    if (!ctx) return -1;
    char *dup = NULL;
    if (prompt && prompt[0] != '\0') {
        dup = strdup(prompt);
        if (!dup) return -1;
    }
    free(ctx->prompt);
    ctx->prompt = dup;

    free(ctx->prompt_tokens);
    ctx->prompt_tokens = NULL;
    ctx->n_prompt_tokens = 0;
    ctx->prompt_tokens_ready = 0;
    return 0;
}

static int prepare_prompt_tokens(q25_ctx_t *ctx, qwen_tokenizer_t *tokenizer) {
    if (ctx->prompt_tokens_ready) return 0;

    free(ctx->prompt_tokens);
    ctx->prompt_tokens = NULL;
    ctx->n_prompt_tokens = 0;

    if (ctx->prompt && ctx->prompt[0] != '\0') {
        ctx->prompt_tokens = qwen_tokenizer_encode(tokenizer, ctx->prompt, &ctx->n_prompt_tokens);
        if (!ctx->prompt_tokens) {
            fprintf(stderr, "q25: failed to encode --prompt text\n");
            return -1;
        }
    }

    ctx->prompt_tokens_ready = 1;
    return 0;
}

/* ========================================================================
 * Prompt Template Token Arrays
 *
 * Same im_start/im_end/endoftext as Qwen3 (151644/151645/151643)
 * but audio_start = 151647, audio_end = 151648
 * ======================================================================== */

static const int Q25_PREFIX_HEAD[] = {
    151644, 8948, 198               /* <|im_start|>system\n */
};
static const int Q25_PREFIX_TAIL[] = {
    151645, 198, 151644, 872, 198, 151647  /* <|im_end|>\n<|im_start|>user\n<audio_start> */
};
static const int Q25_SUFFIX_BASE[] = {
    151648, 151645, 198, 151644, 77091, 198  /* <audio_end><|im_end|>\n<|im_start|>assistant\n */
};

static const int Q25_USER_HEAD[] = {
    151645, 198, 151644, 872, 198   /* <|im_end|>\n<|im_start|>user\n */
};
static const int Q25_USER_TAIL[] = {
    151645, 198, 151644, 77091, 198 /* <|im_end|>\n<|im_start|>assistant\n */
};

#define Q25_PREFIX_HEAD_LEN 3
#define Q25_PREFIX_TAIL_LEN 6
#define Q25_SUFFIX_BASE_LEN 6
#define Q25_USER_HEAD_LEN 5
#define Q25_USER_TAIL_LEN 5

/* Convert a single token embedding from bf16 to f32 */
static void tok_embed_bf16_to_f32(float *dst, const uint16_t *tok_emb_bf16,
                                  int token_id, int dim) {
    const uint16_t *src = tok_emb_bf16 + (size_t)token_id * dim;
    for (int i = 0; i < dim; i++) {
        uint32_t f32_bits = ((uint32_t)src[i]) << 16;
        memcpy(&dst[i], &f32_bits, sizeof(float));
    }
}

/* ========================================================================
 * Token Sampling
 * ======================================================================== */

static int sample_token(float *logits, int vocab_size,
                        const int *recent_tokens, int n_recent,
                        float temperature, float repetition_penalty, int top_k) {
    /* 1. Repetition penalty */
    if (repetition_penalty != 1.0f && n_recent > 0) {
        for (int i = 0; i < n_recent; i++) {
            int tid = recent_tokens[i];
            if (tid < 0 || tid >= vocab_size) continue;
            if (logits[tid] > 0.0f)
                logits[tid] /= repetition_penalty;
            else
                logits[tid] *= repetition_penalty;
        }
    }

    /* 2. Temperature scaling */
    if (temperature > 0.0f && temperature != 1.0f) {
        for (int i = 0; i < vocab_size; i++)
            logits[i] /= temperature;
    }

    /* 3. Top-k filtering */
    if (top_k > 0 && top_k < vocab_size) {
        float *topk_vals = (float *)malloc(top_k * sizeof(float));
        for (int i = 0; i < top_k; i++) topk_vals[i] = -1e30f;
        float topk_min = -1e30f;
        int topk_min_idx = 0;
        for (int i = 0; i < vocab_size; i++) {
            if (logits[i] > topk_min) {
                topk_vals[topk_min_idx] = logits[i];
                topk_min = topk_vals[0];
                topk_min_idx = 0;
                for (int j = 1; j < top_k; j++) {
                    if (topk_vals[j] < topk_min) {
                        topk_min = topk_vals[j];
                        topk_min_idx = j;
                    }
                }
            }
        }
        float kth = topk_min;
        free(topk_vals);

        for (int i = 0; i < vocab_size; i++) {
            if (logits[i] < kth) logits[i] = -1e30f;
        }
    }

    /* 4. Softmax */
    qwen_softmax(logits, 1, vocab_size);

    /* 5. Multinomial sampling */
    double r = drand48();
    double cumsum = 0.0;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += (double)logits[i];
        if (cumsum >= r) return i;
    }
    return vocab_size - 1;
}

/* ========================================================================
 * Model Loading
 * ======================================================================== */

q25_ctx_t *q25_load(const char *model_dir) {
    q25_ctx_t *ctx = (q25_ctx_t *)calloc(1, sizeof(q25_ctx_t));
    if (!ctx) return NULL;
    snprintf(ctx->model_dir, sizeof(ctx->model_dir), "%s", model_dir);

    /* Open safetensors (multi-shard) */
    if (qwen_verbose >= 1)
        fprintf(stderr, "Loading model from %s\n", model_dir);

    multi_safetensors_t *ms = multi_safetensors_open(model_dir);
    if (!ms) {
        fprintf(stderr, "q25_load: cannot open safetensors in %s\n", model_dir);
        free(ctx);
        return NULL;
    }
    ctx->safetensors = ms;

    /* Hardcoded config from Qwen2.5-Omni-7B config.json */
    q25_config_t *cfg = &ctx->config;
    cfg->enc_d_model = 1280;
    cfg->enc_layers = 32;
    cfg->enc_heads = 20;
    cfg->enc_head_dim = 64;
    cfg->enc_ffn_dim = 5120;
    cfg->enc_output_dim = 3584;
    cfg->enc_n_window = 100;

    cfg->dec_hidden = 3584;
    cfg->dec_layers = 28;
    cfg->dec_heads = 28;
    cfg->dec_kv_heads = 4;
    cfg->dec_head_dim = 128;
    cfg->dec_intermediate = 18944;
    cfg->vocab_size = Q25_VOCAB_SIZE;
    cfg->dec_rms_norm_eps = 1e-6f;
    cfg->dec_rope_theta = 1e6f;

    if (qwen_verbose >= 1) fprintf(stderr, "Detected: Qwen2.5-Omni-7B\n");

    /* Load encoder weights */
    if (qwen_verbose >= 1) fprintf(stderr, "Loading encoder weights...\n");
    if (q25_encoder_load(&ctx->encoder, ms, cfg) != 0) {
        fprintf(stderr, "q25_load: failed to load encoder\n");
        q25_free(ctx);
        return NULL;
    }

    /* Load decoder weights */
    if (qwen_verbose >= 1) fprintf(stderr, "Loading decoder weights...\n");
    if (q25_decoder_load(&ctx->decoder, ms, cfg) != 0) {
        fprintf(stderr, "q25_load: failed to load decoder\n");
        q25_free(ctx);
        return NULL;
    }

    /* Defaults */
    ctx->thinker_mode = 0;
    ctx->thinker_max_tokens = 2048;
    ctx->temperature = 0.7f;
    ctx->repetition_penalty = 1.1f;
    ctx->top_k = 40;

    if (qwen_verbose >= 1) fprintf(stderr, "Model loaded.\n");
    return ctx;
}

/* ========================================================================
 * Free
 * ======================================================================== */

void q25_free(q25_ctx_t *ctx) {
    if (!ctx) return;

    #define FREE0(p) do { free(p); (p) = NULL; } while (0)

    /* Encoder conv stem */
    FREE0(ctx->encoder.conv1_weight); FREE0(ctx->encoder.conv1_bias);
    FREE0(ctx->encoder.conv2_weight); FREE0(ctx->encoder.conv2_bias);

    /* Encoder layers */
    for (int i = 0; i < ctx->config.enc_layers; i++) {
        q25_enc_layer_t *l = &ctx->encoder.layers[i];
        FREE0(l->wq_weight); FREE0(l->wq_bias);
        FREE0(l->wk_weight);
        FREE0(l->wv_weight); FREE0(l->wv_bias);
        FREE0(l->wo_weight); FREE0(l->wo_bias);
        FREE0(l->attn_norm_weight); FREE0(l->attn_norm_bias);
        FREE0(l->fc1_weight); FREE0(l->fc1_bias);
        FREE0(l->fc2_weight); FREE0(l->fc2_bias);
        FREE0(l->ffn_norm_weight); FREE0(l->ffn_norm_bias);
    }
    FREE0(ctx->encoder.ln_post_weight); FREE0(ctx->encoder.ln_post_bias);
    FREE0(ctx->encoder.proj_weight); FREE0(ctx->encoder.proj_bias);
    FREE0(ctx->encoder.audio_bos_eos);

    /* Decoder layers */
    for (int i = 0; i < ctx->config.dec_layers; i++) {
        q25_dec_layer_t *l = &ctx->decoder.layers[i];
        FREE0(l->wq_bias); FREE0(l->wk_bias); FREE0(l->wv_bias);
        FREE0(l->input_norm); FREE0(l->post_attn_norm);
        FREE0(l->gate_up_fused_bf16);
    }
    FREE0(ctx->decoder.norm);

    #undef FREE0

    /* KV cache */
    free(ctx->kv_cache_k);
    free(ctx->kv_cache_v);

    /* Decoder buffers */
    free(ctx->dec_x); free(ctx->dec_x_norm);
    free(ctx->dec_q); free(ctx->dec_k); free(ctx->dec_v);
    free(ctx->dec_attn_out); free(ctx->dec_proj_out);
    free(ctx->dec_gate); free(ctx->dec_ffn_out);

    /* Prefill buffers */
    free(ctx->pref_x); free(ctx->pref_x_norm);
    free(ctx->pref_q); free(ctx->pref_k); free(ctx->pref_v);
    free(ctx->pref_attn_out); free(ctx->pref_proj_out); free(ctx->pref_ffn_out);
    free(ctx->pref_gate); free(ctx->pref_gate_up);

    /* RoPE caches */
    free(ctx->rope_cache_cos); free(ctx->rope_cache_sin);
    free(ctx->rope_inv_freq);

    /* Prompt */
    free(ctx->prompt);
    free(ctx->prompt_tokens);

    /* Close safetensors */
    if (ctx->safetensors) {
        multi_safetensors_close((multi_safetensors_t *)ctx->safetensors);
    }

    free(ctx);
}

/* ========================================================================
 * Thinker Generate
 * ======================================================================== */

char *q25_thinker_generate(q25_ctx_t *ctx, const float *samples, int n_samples,
                            const char *user_text) {
    if (!ctx) return NULL;
    if (!samples && !user_text) {
        fprintf(stderr, "q25_thinker: need audio samples or user text\n");
        return NULL;
    }

    const q25_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int max_tokens = ctx->thinker_max_tokens > 0 ? ctx->thinker_max_tokens : 2048;

    ctx->perf_total_ms = 0;
    ctx->perf_text_tokens = 0;
    ctx->perf_audio_ms = samples ? 1000.0 * (double)n_samples / 16000.0 : 0;
    ctx->perf_encode_ms = 0;
    ctx->perf_decode_ms = 0;

    double total_t0 = get_time_ms();

    /* Load tokenizer */
    char vocab_path[1024];
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", ctx->model_dir);
    qwen_tokenizer_t *tokenizer = qwen_tokenizer_load(vocab_path);
    if (!tokenizer) return NULL;

    if (prepare_prompt_tokens(ctx, tokenizer) != 0) {
        qwen_tokenizer_free(tokenizer);
        return NULL;
    }

    /* ---- Encode audio (if provided) ---- */
    float *enc_output = NULL;
    int enc_seq_len = 0;
    double encode_ms = 0;

    if (samples && n_samples > 0) {
        double t0 = get_time_ms();
        int mel_frames = 0;
        float *mel = qwen_mel_spectrogram(samples, n_samples, &mel_frames);
        if (!mel) { qwen_tokenizer_free(tokenizer); return NULL; }

        enc_output = q25_encoder_forward(ctx, mel, mel_frames, &enc_seq_len);
        free(mel);
        if (!enc_output) { qwen_tokenizer_free(tokenizer); return NULL; }
        encode_ms = get_time_ms() - t0;

        if (qwen_verbose >= 2)
            fprintf(stderr, "  Q25 encoder: %d tokens (%.0f ms)\n", enc_seq_len, encode_ms);
    }

    /* ---- Encode user text (if provided) ---- */
    int *user_tokens = NULL;
    int n_user_tokens = 0;
    if (user_text && user_text[0] != '\0') {
        user_tokens = qwen_tokenizer_encode(tokenizer, user_text, &n_user_tokens);
        if (!user_tokens) {
            fprintf(stderr, "q25_thinker: failed to encode user text\n");
            free(enc_output);
            qwen_tokenizer_free(tokenizer);
            return NULL;
        }
    }

    /* ---- Build input embeddings ---- */
    int total_seq;
    float *input_embeds = NULL;

    if (enc_output) {
        /* Audio path */
        int prefix_len = Q25_PREFIX_HEAD_LEN + ctx->n_prompt_tokens + Q25_PREFIX_TAIL_LEN;
        total_seq = prefix_len + enc_seq_len + Q25_SUFFIX_BASE_LEN;

        input_embeds = (float *)malloc((size_t)total_seq * dim * sizeof(float));
        if (!input_embeds) {
            free(enc_output); free(user_tokens);
            qwen_tokenizer_free(tokenizer);
            return NULL;
        }

        int off = 0;
        for (int i = 0; i < Q25_PREFIX_HEAD_LEN; i++) {
            tok_embed_bf16_to_f32(input_embeds + off * dim,
                                  ctx->decoder.tok_embeddings_bf16,
                                  Q25_PREFIX_HEAD[i], dim);
            off++;
        }
        for (int i = 0; i < ctx->n_prompt_tokens; i++) {
            tok_embed_bf16_to_f32(input_embeds + off * dim,
                                  ctx->decoder.tok_embeddings_bf16,
                                  ctx->prompt_tokens[i], dim);
            off++;
        }
        for (int i = 0; i < Q25_PREFIX_TAIL_LEN; i++) {
            tok_embed_bf16_to_f32(input_embeds + off * dim,
                                  ctx->decoder.tok_embeddings_bf16,
                                  Q25_PREFIX_TAIL[i], dim);
            off++;
        }
        /* Audio encoder embeddings (already in output_dim = dec_hidden = 3584) */
        memcpy(input_embeds + (size_t)prefix_len * dim,
               enc_output, (size_t)enc_seq_len * dim * sizeof(float));
        free(enc_output);

        int suffix_off = prefix_len + enc_seq_len;
        for (int i = 0; i < Q25_SUFFIX_BASE_LEN; i++) {
            tok_embed_bf16_to_f32(input_embeds + (suffix_off + i) * dim,
                                  ctx->decoder.tok_embeddings_bf16,
                                  Q25_SUFFIX_BASE[i], dim);
        }
    } else {
        /* Text-only path */
        total_seq = Q25_PREFIX_HEAD_LEN + ctx->n_prompt_tokens
                  + Q25_USER_HEAD_LEN + n_user_tokens + Q25_USER_TAIL_LEN;

        input_embeds = (float *)malloc((size_t)total_seq * dim * sizeof(float));
        if (!input_embeds) {
            free(user_tokens);
            qwen_tokenizer_free(tokenizer);
            return NULL;
        }

        int off = 0;
        for (int i = 0; i < Q25_PREFIX_HEAD_LEN; i++) {
            tok_embed_bf16_to_f32(input_embeds + off * dim,
                                  ctx->decoder.tok_embeddings_bf16,
                                  Q25_PREFIX_HEAD[i], dim);
            off++;
        }
        for (int i = 0; i < ctx->n_prompt_tokens; i++) {
            tok_embed_bf16_to_f32(input_embeds + off * dim,
                                  ctx->decoder.tok_embeddings_bf16,
                                  ctx->prompt_tokens[i], dim);
            off++;
        }
        for (int i = 0; i < Q25_USER_HEAD_LEN; i++) {
            tok_embed_bf16_to_f32(input_embeds + off * dim,
                                  ctx->decoder.tok_embeddings_bf16,
                                  Q25_USER_HEAD[i], dim);
            off++;
        }
        for (int i = 0; i < n_user_tokens; i++) {
            tok_embed_bf16_to_f32(input_embeds + off * dim,
                                  ctx->decoder.tok_embeddings_bf16,
                                  user_tokens[i], dim);
            off++;
        }
        for (int i = 0; i < Q25_USER_TAIL_LEN; i++) {
            tok_embed_bf16_to_f32(input_embeds + off * dim,
                                  ctx->decoder.tok_embeddings_bf16,
                                  Q25_USER_TAIL[i], dim);
            off++;
        }
    }
    free(user_tokens);

    /* ---- Sampling parameters ---- */
    float temperature = ctx->temperature;
    float rep_penalty = ctx->repetition_penalty;
    int top_k_param = ctx->top_k;
    int use_sampling = (temperature > 0.0f);

    if (use_sampling) srand48((long)get_time_ms());

    /* ---- Decoder prefill ---- */
    double t0 = get_time_ms();
    ctx->kv_cache_len = 0;
    int prefill_len = total_seq - 1;
    q25_decoder_prefill(ctx, input_embeds, prefill_len);

    float *logits = (float *)malloc((size_t)cfg->vocab_size * sizeof(float));
    int rep_window = 64;
    int *recent_tokens = (int *)malloc(rep_window * sizeof(int));
    int n_recent = 0;

    /* First token from last prefill position */
    float *last_embed = input_embeds + (size_t)prefill_len * dim;
    int token;
    if (use_sampling) {
        q25_decoder_forward_logits(ctx, last_embed, logits);
        token = sample_token(logits, cfg->vocab_size, recent_tokens, n_recent,
                             temperature, rep_penalty, top_k_param);
    } else {
        token = q25_decoder_forward(ctx, last_embed);
    }
    free(input_embeds);

    double prefill_ms = get_time_ms() - t0;
    if (qwen_verbose >= 2)
        fprintf(stderr, "  Q25 prefill: %d tokens (%.0f ms)\n", total_seq, prefill_ms);

    /* ---- Autoregressive decode ---- */
    t0 = get_time_ms();
    float *tmp_embed = (float *)malloc(dim * sizeof(float));
    if (!tmp_embed) {
        free(logits); free(recent_tokens);
        qwen_tokenizer_free(tokenizer); return NULL;
    }

    size_t text_cap = 4096;
    size_t text_len = 0;
    char *text = (char *)malloc(text_cap);
    text[0] = '\0';
    int n_text_tokens = 0;

    for (int i = 0; i < max_tokens; i++) {
        if (token == Q25_ENDOFTEXT || token == Q25_IM_END) break;

        /* Track recent tokens for repetition penalty */
        if (n_recent < rep_window) {
            recent_tokens[n_recent++] = token;
        } else {
            memmove(recent_tokens, recent_tokens + 1, (rep_window - 1) * sizeof(int));
            recent_tokens[rep_window - 1] = token;
        }

        const char *piece = qwen_tokenizer_decode(tokenizer, token);
        if (piece && piece[0] != '\0') {
            size_t piece_len = strlen(piece);
            if (text_len + piece_len + 1 > text_cap) {
                while (text_len + piece_len + 1 > text_cap) text_cap *= 2;
                text = (char *)realloc(text, text_cap);
            }
            memcpy(text + text_len, piece, piece_len);
            text_len += piece_len;
            text[text_len] = '\0';
            n_text_tokens++;

            if (ctx->token_cb)
                ctx->token_cb(piece, ctx->token_cb_userdata);
        }

        /* Embed and generate next token */
        tok_embed_bf16_to_f32(tmp_embed, ctx->decoder.tok_embeddings_bf16, token, dim);
        if (use_sampling) {
            q25_decoder_forward_logits(ctx, tmp_embed, logits);
            token = sample_token(logits, cfg->vocab_size, recent_tokens, n_recent,
                                 temperature, rep_penalty, top_k_param);
        } else {
            token = q25_decoder_forward(ctx, tmp_embed);
        }
    }

    double decode_ms = get_time_ms() - t0;
    free(tmp_embed);
    free(logits);
    free(recent_tokens);
    qwen_tokenizer_free(tokenizer);

    if (qwen_verbose >= 2)
        fprintf(stderr, "  Q25 decode: %d tokens (%.0f ms, %.1f ms/token)\n",
                n_text_tokens, decode_ms,
                n_text_tokens > 0 ? decode_ms / n_text_tokens : 0);

    /* ---- Perf stats ---- */
    double total_ms = get_time_ms() - total_t0;
    ctx->perf_total_ms = total_ms;
    ctx->perf_text_tokens = n_text_tokens;
    ctx->perf_encode_ms = encode_ms;
    ctx->perf_decode_ms = prefill_ms + decode_ms;

    return text;
}

/* ========================================================================
 * ASR-style Transcription
 * ======================================================================== */

char *q25_transcribe_audio(q25_ctx_t *ctx, const float *samples, int n_samples) {
    if (!ctx || !samples || n_samples <= 0) return NULL;

    /* For ASR, use a transcription-oriented system prompt */
    int had_prompt = (ctx->prompt != NULL);
    if (!had_prompt) {
        q25_set_prompt(ctx, "You are a helpful assistant.");
    }

    /* Use thinker generate with audio, no user text */
    char *text = q25_thinker_generate(ctx, samples, n_samples, NULL);

    if (!had_prompt) {
        q25_set_prompt(ctx, NULL);
    }

    return text;
}
