/*
 * smolvlm_image.c - Image loading via stb_image, resize, and normalization
 *
 * Supports PNG, JPG, BMP, GIF, PSD, TGA, HDR, PIC, PNM (PPM/PGM) via stb_image.
 * Output: channel-first float [3, H, W] normalized to [-1, 1].
 */

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR          /* skip HDR loader (we don't need float images) */
#define STBI_NO_LINEAR       /* skip linear float conversion */
#include "stb_image.h"

#include "smolvlm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * Bilinear Resize
 * ======================================================================== */

static float *bilinear_resize(const unsigned char *src, int src_w, int src_h,
                               int dst_w, int dst_h) {
    /* Output: [dst_h, dst_w, 3] float32 in [0, 255] range */
    float *dst = (float *)malloc((size_t)dst_h * dst_w * 3 * sizeof(float));
    if (!dst) return NULL;

    for (int y = 0; y < dst_h; y++) {
        float src_y = (float)y * (src_h - 1) / (dst_h > 1 ? dst_h - 1 : 1);
        int y0 = (int)src_y;
        int y1 = y0 + 1;
        if (y1 >= src_h) y1 = src_h - 1;
        float fy = src_y - y0;

        for (int x = 0; x < dst_w; x++) {
            float src_x = (float)x * (src_w - 1) / (dst_w > 1 ? dst_w - 1 : 1);
            int x0 = (int)src_x;
            int x1 = x0 + 1;
            if (x1 >= src_w) x1 = src_w - 1;
            float fx = src_x - x0;

            for (int c = 0; c < 3; c++) {
                float v00 = src[(y0 * src_w + x0) * 3 + c];
                float v01 = src[(y0 * src_w + x1) * 3 + c];
                float v10 = src[(y1 * src_w + x0) * 3 + c];
                float v11 = src[(y1 * src_w + x1) * 3 + c];
                float val = v00 * (1 - fy) * (1 - fx)
                          + v10 * fy * (1 - fx)
                          + v01 * (1 - fy) * fx
                          + v11 * fy * fx;
                dst[(y * dst_w + x) * 3 + c] = val;
            }
        }
    }
    return dst;
}

/* ========================================================================
 * Public API
 * ======================================================================== */

float *smolvlm_load_image(const char *path, int target_size, int *out_w, int *out_h) {
    int w, h, channels;
    unsigned char *pixels = stbi_load(path, &w, &h, &channels, 3);  /* force RGB */
    if (!pixels) {
        fprintf(stderr, "smolvlm_image: cannot load %s: %s\n", path, stbi_failure_reason());
        return NULL;
    }

    if (qwen_verbose >= 2) {
        fprintf(stderr, "  Image: %dx%d (%d channels) -> %dx%d\n",
                w, h, channels, target_size, target_size);
    }

    /* Bilinear resize to target_size x target_size */
    float *resized = bilinear_resize(pixels, w, h, target_size, target_size);
    stbi_image_free(pixels);
    if (!resized) return NULL;

    /* Convert to channel-first [3, target_size, target_size] and normalize to [-1, 1] */
    int n = target_size * target_size;
    float *output = (float *)malloc(3 * n * sizeof(float));
    if (!output) {
        free(resized);
        return NULL;
    }

    for (int y = 0; y < target_size; y++) {
        for (int x = 0; x < target_size; x++) {
            int idx = y * target_size + x;
            for (int c = 0; c < 3; c++) {
                float val = resized[idx * 3 + c] / 255.0f * 2.0f - 1.0f;
                output[c * n + idx] = val;
            }
        }
    }

    free(resized);
    *out_w = target_size;
    *out_h = target_size;
    return output;
}
