/*
 * smolvlm_main.c - CLI entry point for SmolVLM-Instruct
 *
 * Usage: smolvlm -d <model_dir> -i <image.pnm> -p <prompt> [options]
 */

#include "smolvlm.h"
#include "common_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Token streaming callback: print each piece as it's decoded */
static void stream_token(const char *piece, void *userdata) {
    (void)userdata;
    fputs(piece, stdout);
    fflush(stdout);
}

static void usage(const char *prog) {
    fprintf(stderr, "smolvlm — SmolVLM-Instruct vision-language inference (pure C)\n\n");
    fprintf(stderr, "Usage: %s -d <model_dir> -i <image.pnm> [options]\n\n", prog);
    fprintf(stderr, "Required:\n");
    fprintf(stderr, "  -d <dir>          Model directory (with *.safetensors, tokenizer.json)\n");
    fprintf(stderr, "  -i <file>         Input image (PNG, JPG, BMP, PNM, TGA, GIF, PSD)\n");
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  -p <text>         Text prompt (default: \"Describe this image.\")\n");
    fprintf(stderr, "  --system-prompt <text>  System prompt for instruct model\n");
    fprintf(stderr, "  -t <n>            Number of threads (default: all CPUs)\n");
    fprintf(stderr, "  --max-tokens <n>  Maximum tokens to generate (default: 256)\n");
    fprintf(stderr, "  --debug           Verbose debug output\n");
    fprintf(stderr, "  --silent          No status output (only generated text on stdout)\n");
    fprintf(stderr, "  -h                Show this help\n");
}

int main(int argc, char **argv) {
    const char *model_dir = NULL;
    const char *image_path = NULL;
    const char *prompt = "Describe this image.";
    const char *system_prompt = NULL;
    int verbosity = 1;
    int n_threads = 0;
    int max_tokens = 256;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            image_path = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--system-prompt") == 0 && i + 1 < argc) {
            system_prompt = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--debug") == 0) {
            verbosity = 2;
        } else if (strcmp(argv[i], "--silent") == 0) {
            verbosity = 0;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!model_dir || !image_path) {
        usage(argv[0]);
        return 1;
    }

    qwen_verbose = verbosity;

    /* Initialize thread pool */
    if (n_threads <= 0) n_threads = qwen_get_num_cpus();
    qwen_set_threads(n_threads);

    /* Load model */
    smolvlm_ctx_t *ctx = smolvlm_load(model_dir);
    if (!ctx) {
        fprintf(stderr, "Failed to load model from %s\n", model_dir);
        return 1;
    }

    /* Set up token streaming */
    int emit_tokens = (verbosity > 0);
    if (emit_tokens)
        smolvlm_set_token_callback(ctx, stream_token, NULL);

    /* Generate */
    char *text = smolvlm_generate(ctx, image_path, prompt, system_prompt, max_tokens);

    if (text) {
        if (emit_tokens)
            printf("\n");
        else
            printf("%s\n", text);
        free(text);
    } else {
        fprintf(stderr, "Generation failed\n");
        smolvlm_free(ctx);
        return 1;
    }

    /* Performance summary */
    if (verbosity >= 1) {
        double tokens_per_sec = 0.0;
        if (ctx->perf_total_ms > 0) {
            tokens_per_sec = (1000.0 * (double)ctx->perf_tokens) / ctx->perf_total_ms;
        }
        fprintf(stderr,
                "Inference: %.0f ms, %d tokens (%.2f tok/s, encoding: %.0fms, decoding: %.0fms)\n",
                ctx->perf_total_ms, ctx->perf_tokens, tokens_per_sec,
                ctx->perf_encode_ms, ctx->perf_decode_ms);
    }

    smolvlm_free(ctx);
    return 0;
}
