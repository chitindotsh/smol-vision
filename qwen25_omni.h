/*
 * qwen25_omni.h - Qwen2.5-Omni-7B types and API
 *
 * Separate model family from Qwen3-ASR/Omni:
 *   - Conv1D encoder stem (not Conv2D)
 *   - Encoder K has no bias
 *   - Single proj (not proj1+proj2)
 *   - Learned audio_bos_eos_token embeddings
 *   - Decoder QKV have biases (O does not)
 *   - No per-head Q/K RMSNorm in decoder
 *   - Separate lm_head (not tied embeddings)
 *   - vocab_size = 152064
 *   - audio_start = 151647, audio_end = 151648
 */

#ifndef QWEN25_OMNI_H
#define QWEN25_OMNI_H

#include <stddef.h>
#include <stdint.h>

/* Token callback type (matches qwen_asr.h definition) */
#ifndef QWEN_ASR_H
typedef void (*qwen_token_cb)(const char *piece, void *userdata);
#endif

/* ========================================================================
 * Constants
 * ======================================================================== */

#define Q25_VOCAB_SIZE        152064
#define Q25_AUDIO_START       151647
#define Q25_AUDIO_END         151648
#define Q25_AUDIO_TOKEN       151646   /* audio pad/placeholder */
#define Q25_IM_START          151644
#define Q25_IM_END            151645
#define Q25_ENDOFTEXT         151643

#define Q25_MAX_ENC_LAYERS    32
#define Q25_MAX_DEC_LAYERS    32

/* ========================================================================
 * Model Configuration
 * ======================================================================== */

typedef struct {
    /* Audio encoder */
    int enc_d_model;           /* 1280 */
    int enc_layers;            /* 32 */
    int enc_heads;             /* 20 */
    int enc_head_dim;          /* 64 */
    int enc_ffn_dim;           /* 5120 */
    int enc_output_dim;        /* 3584 */
    int enc_n_window;          /* 100 (tokens) */

    /* LLM decoder */
    int dec_hidden;            /* 3584 */
    int dec_layers;            /* 28 */
    int dec_heads;             /* 28 */
    int dec_kv_heads;          /* 4 */
    int dec_head_dim;          /* 128 */
    int dec_intermediate;      /* 18944 */
    int vocab_size;            /* 152064 */
    float dec_rms_norm_eps;    /* 1e-6 */
    float dec_rope_theta;      /* 1e6 */
} q25_config_t;

/* ========================================================================
 * Audio Encoder
 * ======================================================================== */

typedef struct {
    /* Self-attention (Q/V have biases, K does NOT, O has bias) */
    float *wq_weight;          /* [d_model, d_model] */
    float *wq_bias;            /* [d_model] */
    float *wk_weight;          /* [d_model, d_model] */
    /* NO wk_bias */
    float *wv_weight;          /* [d_model, d_model] */
    float *wv_bias;            /* [d_model] */
    float *wo_weight;          /* [d_model, d_model] */
    float *wo_bias;            /* [d_model] */

    /* Pre-attention LayerNorm */
    float *attn_norm_weight;   /* [d_model] */
    float *attn_norm_bias;     /* [d_model] */

    /* FFN: GELU(fc1(x)) -> fc2 */
    float *fc1_weight;         /* [ffn_dim, d_model] */
    float *fc1_bias;           /* [ffn_dim] */
    float *fc2_weight;         /* [d_model, ffn_dim] */
    float *fc2_bias;           /* [d_model] */

    /* Pre-FFN LayerNorm */
    float *ffn_norm_weight;    /* [d_model] */
    float *ffn_norm_bias;      /* [d_model] */
} q25_enc_layer_t;

typedef struct {
    /* Conv1D stem (2 layers) */
    float *conv1_weight;       /* [1280, 128, 3] */
    float *conv1_bias;         /* [1280] */
    float *conv2_weight;       /* [1280, 1280, 3] */
    float *conv2_bias;         /* [1280] */

    /* Transformer layers */
    q25_enc_layer_t layers[Q25_MAX_ENC_LAYERS];

    /* Final LayerNorm */
    float *ln_post_weight;     /* [d_model] */
    float *ln_post_bias;       /* [d_model] */

    /* Single projection */
    float *proj_weight;        /* [output_dim, d_model] */
    float *proj_bias;          /* [output_dim] */

    /* Learned boundary tokens */
    float *audio_bos_eos;      /* [2, output_dim] — row 0=bos, row 1=eos */
} q25_encoder_t;

/* ========================================================================
 * LLM Decoder
 * ======================================================================== */

typedef struct {
    /* Self-attention: Q/K/V have biases, O does not */
    uint16_t *wq_weight_bf16;  /* [n_heads*head_dim, hidden] */
    float *wq_bias;            /* [n_heads*head_dim] */
    uint16_t *wk_weight_bf16;  /* [n_kv_heads*head_dim, hidden] */
    float *wk_bias;            /* [n_kv_heads*head_dim] */
    uint16_t *wv_weight_bf16;  /* [n_kv_heads*head_dim, hidden] */
    float *wv_bias;            /* [n_kv_heads*head_dim] */
    uint16_t *wo_weight_bf16;  /* [hidden, n_heads*head_dim] */
    /* NO wo_bias, NO q_norm, NO k_norm */

    /* RMSNorm (no bias) */
    float *input_norm;         /* [hidden] */
    float *post_attn_norm;     /* [hidden] */

    /* SwiGLU MLP (no biases) */
    uint16_t *gate_weight_bf16; /* [intermediate, hidden] */
    uint16_t *up_weight_bf16;   /* [intermediate, hidden] */
    uint16_t *down_weight_bf16; /* [hidden, intermediate] */

    /* Fused gate+up for single-token [2*intermediate, hidden] */
    uint16_t *gate_up_fused_bf16;
} q25_dec_layer_t;

typedef struct {
    /* Token embeddings */
    uint16_t *tok_embeddings_bf16; /* [vocab_size, hidden] */

    /* Separate lm_head (NOT tied) */
    uint16_t *lm_head_bf16;       /* [vocab_size, hidden] */

    /* Transformer layers */
    q25_dec_layer_t layers[Q25_MAX_DEC_LAYERS];

    /* Final RMSNorm */
    float *norm;               /* [hidden] */
} q25_decoder_t;

/* ========================================================================
 * Main Context
 * ======================================================================== */

typedef struct {
    q25_config_t config;
    q25_encoder_t encoder;
    q25_decoder_t decoder;

    /* Model files (kept open for mmap) */
    void *safetensors;         /* multi_safetensors_t* */
    char model_dir[512];

    /* KV cache for decoder */
    float *kv_cache_k;
    float *kv_cache_v;
    int kv_cache_len;
    int kv_cache_max;

    /* Persistent decoder buffers (single-token) */
    float *dec_x, *dec_x_norm, *dec_q, *dec_k, *dec_v;
    float *dec_attn_out, *dec_proj_out;
    float *dec_gate, *dec_ffn_out;

    /* Persistent decoder prefill buffers */
    float *pref_x, *pref_x_norm, *pref_q, *pref_k, *pref_v;
    float *pref_attn_out, *pref_proj_out, *pref_ffn_out;
    float *pref_gate, *pref_gate_up;
    int pref_seq_cap;

    /* Cached RoPE tables */
    float *rope_cache_cos, *rope_cache_sin;
    float *rope_inv_freq;
    int rope_cache_cap;
    int rope_inv_freq_half;

    /* Token streaming callback */
    qwen_token_cb token_cb;
    void *token_cb_userdata;

    /* Generation settings */
    int thinker_mode;
    int thinker_max_tokens;
    float temperature;
    float repetition_penalty;
    int top_k;

    /* Prompt */
    char *prompt;
    int *prompt_tokens;
    int n_prompt_tokens;
    int prompt_tokens_ready;

    /* Performance stats */
    double perf_total_ms;
    int perf_text_tokens;
    double perf_audio_ms;
    double perf_encode_ms;
    double perf_decode_ms;
} q25_ctx_t;

/* ========================================================================
 * API Functions
 * ======================================================================== */

q25_ctx_t *q25_load(const char *model_dir);
void q25_free(q25_ctx_t *ctx);

void q25_set_token_callback(q25_ctx_t *ctx, qwen_token_cb cb, void *ud);
int q25_set_prompt(q25_ctx_t *ctx, const char *prompt);

char *q25_thinker_generate(q25_ctx_t *ctx, const float *samples, int n_samples,
                            const char *user_text);
char *q25_transcribe_audio(q25_ctx_t *ctx, const float *samples, int n_samples);

/* ========================================================================
 * Internal Functions
 * ======================================================================== */

/* Encoder */
int q25_encoder_load(q25_encoder_t *enc, void *ms, const q25_config_t *cfg);
float *q25_encoder_forward(q25_ctx_t *ctx, const float *mel, int mel_frames,
                            int *out_seq_len);

/* Decoder */
int q25_decoder_load(q25_decoder_t *dec, void *ms, const q25_config_t *cfg);
void q25_decoder_prefill(q25_ctx_t *ctx, const float *input_embeds, int seq_len);
int q25_decoder_forward(q25_ctx_t *ctx, const float *input_embed);
void q25_decoder_forward_logits(q25_ctx_t *ctx, const float *input_embed, float *logits);

#endif /* QWEN25_OMNI_H */
