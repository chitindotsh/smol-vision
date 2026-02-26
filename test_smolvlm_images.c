/*
 * test_smolvlm_images.c - Test image loading pipeline for SmolVLM
 *
 * Verifies that stb_image-based loader handles various formats and sizes,
 * produces correct output dimensions, and normalizes pixel values properly.
 *
 * Build:  gcc -Wall -O2 -o test_smolvlm_images test_smolvlm_images.c smolvlm_image.o -lm
 * Run:    ./test_smolvlm_images [test_images_dir]
 */

#include "smolvlm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>

int qwen_verbose = 0;

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  [%s] ", name); \
} while(0)

#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg) printf("FAIL: %s\n", msg)

/* Test: loading a single image produces correct dimensions */
static void test_load_and_dimensions(const char *path, int target_size) {
    char name[256];
    snprintf(name, sizeof(name), "load %-30s -> %dx%d", path, target_size, target_size);
    TEST(name);

    int w, h;
    float *img = smolvlm_load_image(path, target_size, &w, &h);
    if (!img) {
        FAIL("smolvlm_load_image returned NULL");
        return;
    }
    if (w != target_size || h != target_size) {
        char msg[128];
        snprintf(msg, sizeof(msg), "expected %dx%d, got %dx%d", target_size, target_size, w, h);
        FAIL(msg);
        free(img);
        return;
    }
    free(img);
    PASS();
}

/* Test: pixel values are in [-1, 1] range after normalization */
static void test_normalization(const char *path) {
    TEST("normalization range [-1,1]");

    int w, h;
    float *img = smolvlm_load_image(path, 384, &w, &h);
    if (!img) {
        FAIL("could not load image");
        return;
    }

    int n = 3 * 384 * 384;
    float min_val = img[0], max_val = img[0];
    for (int i = 1; i < n; i++) {
        if (img[i] < min_val) min_val = img[i];
        if (img[i] > max_val) max_val = img[i];
    }
    free(img);

    if (min_val < -1.001f || max_val > 1.001f) {
        char msg[128];
        snprintf(msg, sizeof(msg), "range [%.4f, %.4f] exceeds [-1,1]", min_val, max_val);
        FAIL(msg);
        return;
    }
    PASS();
}

/* Test: channel-first layout [C, H, W] */
static void test_channel_first_layout(const char *path) {
    TEST("channel-first layout [3,H,W]");

    int w, h;
    float *img = smolvlm_load_image(path, 384, &w, &h);
    if (!img) {
        FAIL("could not load image");
        return;
    }

    int n = 384 * 384;
    /* For the solid red PPM: R channel should be ~1.0, G and B should be ~-1.0 */
    /* Check that the three channel planes are separate */
    /* Each plane is n floats, total 3*n */
    float *r_plane = img;
    float *g_plane = img + n;
    float *b_plane = img + 2 * n;

    /* Just verify the planes don't alias (basic sanity) */
    int distinct = 0;
    for (int i = 0; i < 100 && i < n; i++) {
        if (fabsf(r_plane[i] - g_plane[i]) > 0.01f ||
            fabsf(r_plane[i] - b_plane[i]) > 0.01f) {
            distinct++;
        }
    }

    free(img);

    /* For a non-grayscale image, channels should differ */
    if (distinct == 0) {
        FAIL("all sampled pixels have identical RGB (expected diversity)");
        return;
    }
    PASS();
}

/* Test: solid red image has correct channel values */
static void test_solid_red(const char *dir) {
    TEST("solid red PPM channels");

    char path[512];
    snprintf(path, sizeof(path), "%s/solid_red.ppm", dir);

    int w, h;
    float *img = smolvlm_load_image(path, 384, &w, &h);
    if (!img) {
        FAIL("could not load solid_red.ppm");
        return;
    }

    int n = 384 * 384;
    /* R should be ~1.0 (255/255*2-1=1), G and B should be ~-1.0 (0/255*2-1=-1) */
    float r_avg = 0, g_avg = 0, b_avg = 0;
    for (int i = 0; i < n; i++) {
        r_avg += img[i];
        g_avg += img[n + i];
        b_avg += img[2 * n + i];
    }
    r_avg /= n;
    g_avg /= n;
    b_avg /= n;
    free(img);

    if (fabsf(r_avg - 1.0f) > 0.01f || fabsf(g_avg + 1.0f) > 0.01f || fabsf(b_avg + 1.0f) > 0.01f) {
        char msg[256];
        snprintf(msg, sizeof(msg), "R=%.3f (exp 1.0), G=%.3f (exp -1.0), B=%.3f (exp -1.0)",
                 r_avg, g_avg, b_avg);
        FAIL(msg);
        return;
    }
    PASS();
}

/* Test: various image formats load successfully */
static void test_format_variety(const char *dir) {
    DIR *d = opendir(dir);
    if (!d) return;

    struct dirent *entry;
    while ((entry = readdir(d)) != NULL) {
        if (entry->d_name[0] == '.') continue;
        if (strcmp(entry->d_name, "test_images.json") == 0) continue;

        char path[512];
        snprintf(path, sizeof(path), "%s/%s", dir, entry->d_name);
        test_load_and_dimensions(path, 384);
    }
    closedir(d);
}

/* Test: different target sizes work */
static void test_resize_targets(const char *path) {
    int sizes[] = {64, 128, 224, 384, 512};
    for (int i = 0; i < 5; i++) {
        test_load_and_dimensions(path, sizes[i]);
    }
}

int main(int argc, char **argv) {
    const char *test_dir = "test_images";
    if (argc > 1) test_dir = argv[1];

    printf("SmolVLM Image Loading Tests\n");
    printf("===========================\n");
    printf("Test directory: %s\n\n", test_dir);

    /* Check test directory exists */
    DIR *d = opendir(test_dir);
    if (!d) {
        fprintf(stderr, "Error: test directory '%s' not found.\n", test_dir);
        fprintf(stderr, "Run: python3 img_downloader.py -o %s\n", test_dir);
        return 1;
    }
    closedir(d);

    /* 1. Format variety: load every image in the directory */
    printf("1. Format variety (load all images at 384x384):\n");
    test_format_variety(test_dir);
    printf("\n");

    /* 2. Normalization range */
    printf("2. Normalization:\n");
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/gradient_test.ppm", test_dir);
        test_normalization(path);
    }
    printf("\n");

    /* 3. Channel-first layout */
    printf("3. Channel-first layout:\n");
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/gradient_test.ppm", test_dir);
        test_channel_first_layout(path);
    }
    printf("\n");

    /* 4. Solid color correctness */
    printf("4. Solid color channel correctness:\n");
    test_solid_red(test_dir);
    printf("\n");

    /* 5. Resize to various target sizes */
    printf("5. Resize to various target sizes:\n");
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/building.jpg", test_dir);
        test_resize_targets(path);
    }
    printf("\n");

    /* Summary */
    printf("===========================\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
