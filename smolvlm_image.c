/*
 * smolvlm_image.c - PNM image loading, resize, and normalization
 *
 * Supports PPM P6 (binary RGB) format.
 * Output: channel-first float [3, H, W] normalized to [-1, 1].
 */

#include "smolvlm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* ========================================================================
 * PPM P6 Parser
 * ======================================================================== */

static void skip_whitespace_and_comments(FILE *f) {
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (c == '#') {
            /* Skip comment line */
            while ((c = fgetc(f)) != EOF && c != '\n')
                ;
        } else if (!isspace(c)) {
            ungetc(c, f);
            return;
        }
    }
}

static int read_ppm_int(FILE *f) {
    skip_whitespace_and_comments(f);
    int val = 0;
    int c;
    while ((c = fgetc(f)) != EOF && isdigit(c)) {
        val = val * 10 + (c - '0');
    }
    /* The character after the number is whitespace, consumed */
    return val;
}

/*
 * Load PPM P6 file: binary RGB image.
 * Returns pixel data as [H * W * 3] uint8 array (row-major, interleaved RGB).
 */
static unsigned char *load_ppm(const char *path, int *out_w, int *out_h) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "smolvlm_image: cannot open %s\n", path);
        return NULL;
    }

    /* Read magic */
    char magic[3];
    if (fread(magic, 1, 2, f) != 2 || magic[0] != 'P' || magic[1] != '6') {
        fprintf(stderr, "smolvlm_image: %s is not a PPM P6 file (got '%c%c')\n",
                path, magic[0], magic[1]);
        fclose(f);
        return NULL;
    }

    int w = read_ppm_int(f);
    int h = read_ppm_int(f);
    int maxval = read_ppm_int(f);
    /* After maxval, exactly one whitespace character before pixel data */

    if (w <= 0 || h <= 0 || maxval <= 0 || maxval > 65535) {
        fprintf(stderr, "smolvlm_image: invalid PPM header: %dx%d maxval=%d\n", w, h, maxval);
        fclose(f);
        return NULL;
    }

    size_t npixels = (size_t)w * h * 3;
    unsigned char *pixels = (unsigned char *)malloc(npixels);
    if (!pixels) {
        fclose(f);
        return NULL;
    }

    if (maxval <= 255) {
        /* 8-bit per channel */
        if (fread(pixels, 1, npixels, f) != npixels) {
            fprintf(stderr, "smolvlm_image: short read in %s\n", path);
            free(pixels);
            fclose(f);
            return NULL;
        }
    } else {
        /* 16-bit per channel: read and downscale */
        unsigned char *raw = (unsigned char *)malloc(npixels * 2);
        if (!raw || fread(raw, 1, npixels * 2, f) != npixels * 2) {
            free(raw);
            free(pixels);
            fclose(f);
            return NULL;
        }
        for (size_t i = 0; i < npixels; i++) {
            int val16 = (raw[i * 2] << 8) | raw[i * 2 + 1];
            pixels[i] = (unsigned char)(val16 * 255 / maxval);
        }
        free(raw);
    }

    fclose(f);
    *out_w = w;
    *out_h = h;
    return pixels;
}

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
    int w, h;
    unsigned char *pixels = load_ppm(path, &w, &h);
    if (!pixels) return NULL;

    /* Bilinear resize to target_size x target_size */
    float *resized = bilinear_resize(pixels, w, h, target_size, target_size);
    free(pixels);
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
