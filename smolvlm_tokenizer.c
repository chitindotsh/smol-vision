/*
 * smolvlm_tokenizer.c - GPT-2 BPE tokenizer for SmolVLM
 *
 * Loads vocab + merges from tokenizer.json (HuggingFace fast tokenizer format).
 * Uses same GPT-2 byte-level BPE algorithm as qwen_asr_tokenizer.c.
 */

#include "smolvlm_tokenizer.h"
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * GPT-2 Bytes-to-Unicode Mapping
 * ======================================================================== */

static int gpt2_byte_to_unicode[256];
static int gpt2_unicode_to_byte[512];
static int gpt2_mapping_initialized = 0;

static void init_gpt2_mapping(void) {
    if (gpt2_mapping_initialized) return;

    memset(gpt2_unicode_to_byte, -1, sizeof(gpt2_unicode_to_byte));

    int n = 0;
    for (int b = 0; b < 256; b++) {
        int is_normal = 0;
        if (b >= 33 && b <= 126) is_normal = 1;
        if (b >= 161 && b <= 172) is_normal = 1;
        if (b >= 174 && b <= 255) is_normal = 1;

        if (is_normal) gpt2_byte_to_unicode[b] = b;
        else gpt2_byte_to_unicode[b] = 256 + n++;
    }

    for (int b = 0; b < 256; b++) {
        int cp = gpt2_byte_to_unicode[b];
        if (cp < 512) gpt2_unicode_to_byte[cp] = b;
    }

    gpt2_mapping_initialized = 1;
}

static int utf8_encode_cp(int cp, char out[4]) {
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    out[0] = (char)(0xE0 | (cp >> 12));
    out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[2] = (char)(0x80 | (cp & 0x3F));
    return 3;
}

static char *decode_gpt2_token(const char *token_str) {
    init_gpt2_mapping();

    size_t len = strlen(token_str);
    unsigned char *bytes = (unsigned char *)malloc(len + 1);
    if (!bytes) return NULL;

    int byte_count = 0;
    const unsigned char *p = (const unsigned char *)token_str;
    const unsigned char *end = p + len;

    while (p < end) {
        int cp;
        int nbytes;

        if ((*p & 0x80) == 0) {
            cp = *p;
            nbytes = 1;
        } else if ((*p & 0xE0) == 0xC0) {
            cp = (*p & 0x1F) << 6;
            if (p + 1 < end) cp |= (p[1] & 0x3F);
            nbytes = 2;
        } else if ((*p & 0xF0) == 0xE0) {
            cp = (*p & 0x0F) << 12;
            if (p + 1 < end) cp |= (p[1] & 0x3F) << 6;
            if (p + 2 < end) cp |= (p[2] & 0x3F);
            nbytes = 3;
        } else if ((*p & 0xF8) == 0xF0) {
            cp = (*p & 0x07) << 18;
            if (p + 1 < end) cp |= (p[1] & 0x3F) << 12;
            if (p + 2 < end) cp |= (p[2] & 0x3F) << 6;
            if (p + 3 < end) cp |= (p[3] & 0x3F);
            nbytes = 4;
        } else {
            cp = *p;
            nbytes = 1;
        }
        p += nbytes;

        if (cp < 512 && gpt2_unicode_to_byte[cp] >= 0) {
            bytes[byte_count++] = (unsigned char)gpt2_unicode_to_byte[cp];
        } else {
            bytes[byte_count++] = '?';
        }
    }
    bytes[byte_count] = '\0';
    return (char *)bytes;
}

/* ========================================================================
 * String->int hash map (open addressing)
 * ======================================================================== */

typedef struct {
    char *key;
    int value;
} str_int_entry_t;

static uint64_t fnv1a_hash(const char *s) {
    uint64_t h = 1469598103934665603ULL;
    while (*s) {
        h ^= (unsigned char)*s++;
        h *= 1099511628211ULL;
    }
    return h;
}

static int next_pow2(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

static int map_insert(str_int_entry_t *map, int cap, char *key, int value) {
    if (!map || cap <= 0 || !key) return -1;
    int mask = cap - 1;
    int pos = (int)(fnv1a_hash(key) & (uint64_t)mask);
    for (int i = 0; i < cap; i++) {
        int idx = (pos + i) & mask;
        if (!map[idx].key) {
            map[idx].key = key;
            map[idx].value = value;
            return 0;
        }
        if (strcmp(map[idx].key, key) == 0) {
            map[idx].value = value;
            return 0;
        }
    }
    return -1;
}

static int map_get(const str_int_entry_t *map, int cap, const char *key) {
    if (!map || cap <= 0 || !key) return -1;
    int mask = cap - 1;
    int pos = (int)(fnv1a_hash(key) & (uint64_t)mask);
    for (int i = 0; i < cap; i++) {
        int idx = (pos + i) & mask;
        if (!map[idx].key) return -1;
        if (strcmp(map[idx].key, key) == 0) return map[idx].value;
    }
    return -1;
}

/* ========================================================================
 * BPE helpers
 * ======================================================================== */

static int utf8_char_len(unsigned char c) {
    if ((c & 0x80) == 0) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

static int split_utf8_symbols(const char *s, char ***out_syms, int *out_n) {
    *out_syms = NULL;
    *out_n = 0;
    if (!s || !*s) return 0;

    int n = 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; ) {
        int l = utf8_char_len(*p);
        p += l;
        n++;
    }
    if (n <= 0) return 0;

    char **syms = (char **)calloc((size_t)n, sizeof(char *));
    if (!syms) return -1;

    int i = 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; ) {
        int l = utf8_char_len(*p);
        syms[i] = (char *)malloc((size_t)l + 1);
        if (!syms[i]) {
            for (int j = 0; j < i; j++) free(syms[j]);
            free(syms);
            return -1;
        }
        memcpy(syms[i], p, (size_t)l);
        syms[i][l] = '\0';
        p += l;
        i++;
    }

    *out_syms = syms;
    *out_n = n;
    return 0;
}

static char *str_concat2(const char *a, const char *b) {
    size_t la = strlen(a), lb = strlen(b);
    char *r = (char *)malloc(la + lb + 1);
    if (!r) return NULL;
    memcpy(r, a, la);
    memcpy(r + la, b, lb);
    r[la + lb] = '\0';
    return r;
}

static int merge_rank(const struct smolvlm_tokenizer *tok, const char *a, const char *b) {
    if (!tok->merge_map || tok->merge_map_cap <= 0) return INT_MAX;

    size_t la = strlen(a), lb = strlen(b);
    char *pair = (char *)malloc(la + 1 + lb + 1);
    if (!pair) return INT_MAX;
    memcpy(pair, a, la);
    pair[la] = ' ';
    memcpy(pair + la + 1, b, lb);
    pair[la + 1 + lb] = '\0';

    int rank = map_get((const str_int_entry_t *)tok->merge_map, tok->merge_map_cap, pair);
    free(pair);
    return rank >= 0 ? rank : INT_MAX;
}

static int append_id(int **arr, int *n, int *cap, int id) {
    if (*n >= *cap) {
        int new_cap = (*cap == 0) ? 16 : (*cap * 2);
        int *tmp = (int *)realloc(*arr, (size_t)new_cap * sizeof(int));
        if (!tmp) return -1;
        *arr = tmp;
        *cap = new_cap;
    }
    (*arr)[(*n)++] = id;
    return 0;
}

static char *text_to_bpe_unicode(const char *text) {
    init_gpt2_mapping();
    size_t len = strlen(text);
    char *out = (char *)malloc(len * 3 + 1);
    if (!out) return NULL;

    size_t w = 0;
    for (size_t i = 0; i < len; i++) {
        int cp = gpt2_byte_to_unicode[(unsigned char)text[i]];
        char tmp[4];
        int n = utf8_encode_cp(cp, tmp);
        for (int j = 0; j < n; j++) out[w++] = tmp[j];
    }
    out[w] = '\0';
    return out;
}

static int encode_bpe_word(const struct smolvlm_tokenizer *tok, const char *mapped,
                            int **out_ids, int *out_n) {
    *out_ids = NULL;
    *out_n = 0;
    if (!mapped || !*mapped) return 0;

    char **syms = NULL;
    int n_syms = 0;
    if (split_utf8_symbols(mapped, &syms, &n_syms) != 0) return -1;
    if (n_syms <= 0) {
        free(syms);
        return 0;
    }

    while (n_syms > 1) {
        int best_rank = INT_MAX;
        int best_i = -1;
        for (int i = 0; i < n_syms - 1; i++) {
            int r = merge_rank(tok, syms[i], syms[i + 1]);
            if (r < best_rank) {
                best_rank = r;
                best_i = i;
            }
        }
        if (best_i < 0 || best_rank == INT_MAX) break;

        char *merged = str_concat2(syms[best_i], syms[best_i + 1]);
        if (!merged) {
            for (int i = 0; i < n_syms; i++) free(syms[i]);
            free(syms);
            return -1;
        }
        free(syms[best_i]);
        free(syms[best_i + 1]);
        syms[best_i] = merged;
        for (int j = best_i + 1; j < n_syms - 1; j++) syms[j] = syms[j + 1];
        n_syms--;
    }

    int *ids = NULL;
    int n_ids = 0, cap = 0;
    for (int i = 0; i < n_syms; i++) {
        int id = map_get((const str_int_entry_t *)tok->vocab_map, tok->vocab_map_cap, syms[i]);
        if (id < 0) {
            for (int k = 0; k < n_syms; k++) free(syms[k]);
            free(syms);
            free(ids);
            return -1;
        }
        if (append_id(&ids, &n_ids, &cap, id) != 0) {
            for (int k = 0; k < n_syms; k++) free(syms[k]);
            free(syms);
            free(ids);
            return -1;
        }
    }

    for (int i = 0; i < n_syms; i++) free(syms[i]);
    free(syms);

    *out_ids = ids;
    *out_n = n_ids;
    return 0;
}

/* ========================================================================
 * Minimal JSON parser for tokenizer.json
 * ======================================================================== */

static void skip_ws(const char **p) {
    while (**p == ' ' || **p == '\n' || **p == '\r' || **p == '\t') (*p)++;
}

/*
 * Parse a JSON string value. Handles escape sequences including \uXXXX.
 * Returns 0 on success, -1 on error.
 */
static int parse_json_string(const char **p, char *out, size_t max_len) {
    skip_ws(p);
    if (**p != '"') return -1;
    (*p)++;
    size_t i = 0;
    while (**p && **p != '"' && i < max_len - 1) {
        if (**p == '\\') {
            (*p)++;
            if (**p == 'n') out[i++] = '\n';
            else if (**p == 't') out[i++] = '\t';
            else if (**p == '"') out[i++] = '"';
            else if (**p == '\\') out[i++] = '\\';
            else if (**p == '/') out[i++] = '/';
            else if (**p == 'r') out[i++] = '\r';
            else if (**p == 'u') {
                (*p)++;
                unsigned int cp = 0;
                for (int j = 0; j < 4 && **p; j++, (*p)++) {
                    cp <<= 4;
                    char c = **p;
                    if (c >= '0' && c <= '9') cp |= (unsigned)(c - '0');
                    else if (c >= 'a' && c <= 'f') cp |= (unsigned)(c - 'a' + 10);
                    else if (c >= 'A' && c <= 'F') cp |= (unsigned)(c - 'A' + 10);
                }
                if (cp < 0x80 && i + 1 < max_len) {
                    out[i++] = (char)cp;
                } else if (cp < 0x800 && i + 2 < max_len) {
                    out[i++] = (char)(0xC0 | (cp >> 6));
                    out[i++] = (char)(0x80 | (cp & 0x3F));
                } else if (i + 3 < max_len) {
                    out[i++] = (char)(0xE0 | (cp >> 12));
                    out[i++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                    out[i++] = (char)(0x80 | (cp & 0x3F));
                }
                continue;
            } else {
                out[i++] = **p;
            }
        } else {
            out[i++] = **p;
        }
        (*p)++;
    }
    out[i] = '\0';
    if (**p != '"') return -1;
    (*p)++;
    return 0;
}

/* Skip over a JSON value (string, number, object, array, bool, null) */
static void skip_json_value(const char **p) {
    skip_ws(p);
    if (**p == '"') {
        (*p)++;
        while (**p && **p != '"') {
            if (**p == '\\') (*p)++;
            if (**p) (*p)++;
        }
        if (**p == '"') (*p)++;
    } else if (**p == '{') {
        int depth = 1;
        (*p)++;
        while (**p && depth > 0) {
            if (**p == '{') depth++;
            else if (**p == '}') depth--;
            else if (**p == '"') {
                (*p)++;
                while (**p && **p != '"') {
                    if (**p == '\\') (*p)++;
                    if (**p) (*p)++;
                }
            }
            if (**p) (*p)++;
        }
    } else if (**p == '[') {
        int depth = 1;
        (*p)++;
        while (**p && depth > 0) {
            if (**p == '[') depth++;
            else if (**p == ']') depth--;
            else if (**p == '"') {
                (*p)++;
                while (**p && **p != '"') {
                    if (**p == '\\') (*p)++;
                    if (**p) (*p)++;
                }
            }
            if (**p) (*p)++;
        }
    } else {
        /* number, true, false, null */
        while (**p && **p != ',' && **p != '}' && **p != ']'
               && **p != ' ' && **p != '\n' && **p != '\r' && **p != '\t')
            (*p)++;
    }
}

static int64_t parse_json_int(const char **p) {
    skip_ws(p);
    int neg = 0;
    if (**p == '-') { neg = 1; (*p)++; }
    int64_t val = 0;
    while (**p >= '0' && **p <= '9') {
        val = val * 10 + (**p - '0');
        (*p)++;
    }
    return neg ? -val : val;
}

/*
 * Find a named key within a JSON object and position p right after the ':'.
 * The caller should be positioned right after the opening '{'.
 * Returns 0 if found, -1 if not found.
 */
static int find_json_key(const char **p, const char *key) {
    skip_ws(p);
    while (**p && **p != '}') {
        if (**p == ',') { (*p)++; skip_ws(p); continue; }
        if (**p == '}') break;

        char cur_key[256];
        if (parse_json_string(p, cur_key, sizeof(cur_key)) != 0) return -1;
        skip_ws(p);
        if (**p != ':') return -1;
        (*p)++;

        if (strcmp(cur_key, key) == 0) return 0;

        skip_json_value(p);
        skip_ws(p);
    }
    return -1;
}

/* ========================================================================
 * tokenizer.json Parser
 * ======================================================================== */

static int load_tokenizer_json(struct smolvlm_tokenizer *tok, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "smolvlm_tokenizer: cannot open %s\n", path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *json = (char *)malloc((size_t)file_size + 1);
    if (!json || fread(json, 1, (size_t)file_size, f) != (size_t)file_size) {
        fclose(f);
        free(json);
        return -1;
    }
    fclose(f);
    json[file_size] = '\0';

    /* Find "model" key */
    const char *p = json;
    skip_ws(&p);
    if (*p != '{') { free(json); return -1; }
    p++;

    if (find_json_key(&p, "model") != 0) {
        fprintf(stderr, "smolvlm_tokenizer: 'model' key not found\n");
        free(json);
        return -1;
    }

    /* Now p is right after "model": - expect { */
    skip_ws(&p);
    if (*p != '{') { free(json); return -1; }
    p++;

    /* We need to find "vocab" and "merges" inside the model object */
    /* Save position to search for both */
    const char *model_start = p;

    /* ---- Parse vocab ---- */
    p = model_start;
    if (find_json_key(&p, "vocab") != 0) {
        fprintf(stderr, "smolvlm_tokenizer: 'vocab' key not found in model\n");
        free(json);
        return -1;
    }

    skip_ws(&p);
    if (*p != '{') { free(json); return -1; }
    p++;

    /* First pass: count entries and find max ID */
    int max_id = 0;
    int n_entries = 0;
    const char *vocab_start = p;

    while (*p && *p != '}') {
        skip_ws(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') break;

        char key[4096];
        if (parse_json_string(&p, key, sizeof(key)) != 0) break;
        skip_ws(&p);
        if (*p != ':') break;
        p++;
        int64_t id = parse_json_int(&p);
        if (id > max_id) max_id = (int)id;
        n_entries++;
    }

    /* Allocate lookup tables */
    int vocab_size = max_id + 1;
    tok->vocab_size = vocab_size;
    tok->id_to_text = (char **)calloc((size_t)vocab_size, sizeof(char *));
    tok->id_to_bpe = (char **)calloc((size_t)vocab_size, sizeof(char *));
    if (!tok->id_to_text || !tok->id_to_bpe) {
        free(json);
        return -1;
    }

    /* Second pass: populate */
    p = vocab_start;
    while (*p && *p != '}') {
        skip_ws(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') break;

        char key[4096];
        if (parse_json_string(&p, key, sizeof(key)) != 0) break;
        skip_ws(&p);
        if (*p != ':') break;
        p++;
        int64_t id = parse_json_int(&p);

        if (id >= 0 && id < vocab_size) {
            free(tok->id_to_bpe[id]);
            free(tok->id_to_text[id]);
            tok->id_to_bpe[id] = strdup(key);
            tok->id_to_text[id] = decode_gpt2_token(key);
        }
    }

    /* Build vocab hash map */
    tok->vocab_map_cap = next_pow2(n_entries * 2 + 1);
    tok->vocab_map = calloc((size_t)tok->vocab_map_cap, sizeof(str_int_entry_t));
    if (!tok->vocab_map) { free(json); return -1; }
    for (int i = 0; i < vocab_size; i++) {
        if (tok->id_to_bpe[i]) {
            map_insert((str_int_entry_t *)tok->vocab_map, tok->vocab_map_cap,
                       tok->id_to_bpe[i], i);
        }
    }

    /* ---- Parse merges ---- */
    p = model_start;
    if (find_json_key(&p, "merges") != 0) {
        fprintf(stderr, "smolvlm_tokenizer: 'merges' key not found in model\n");
        free(json);
        return -1;
    }

    skip_ws(&p);
    if (*p != '[') { free(json); return -1; }
    p++;

    /* First pass: count merges */
    int n_merges = 0;
    const char *merges_start = p;
    while (*p && *p != ']') {
        skip_ws(&p);
        if (*p == ',') { p++; continue; }
        if (*p == ']') break;
        if (*p == '"') {
            n_merges++;
            p++;
            while (*p && *p != '"') {
                if (*p == '\\') p++;
                if (*p) p++;
            }
            if (*p == '"') p++;
        } else {
            break;
        }
    }

    /* Build merge rank map */
    tok->merge_map_cap = next_pow2(n_merges * 2 + 1);
    tok->merge_map = calloc((size_t)tok->merge_map_cap, sizeof(str_int_entry_t));
    if (!tok->merge_map) { free(json); return -1; }

    /* Second pass: populate */
    p = merges_start;
    int rank = 0;
    while (*p && *p != ']') {
        skip_ws(&p);
        if (*p == ',') { p++; continue; }
        if (*p == ']') break;

        char merge_str[8192];
        if (parse_json_string(&p, merge_str, sizeof(merge_str)) != 0) break;

        /* merge_str is "token1 token2" - use as merge key directly */
        char *key = strdup(merge_str);
        if (key) {
            if (map_insert((str_int_entry_t *)tok->merge_map, tok->merge_map_cap,
                          key, rank) != 0) {
                free(key);
            }
        }
        rank++;
    }

    /* ---- Handle added_tokens from the top-level ---- */
    /* Reset to top of JSON to find "added_tokens" */
    p = json;
    skip_ws(&p);
    if (*p == '{') {
        p++;
        const char *top_start = p;
        p = top_start;
        if (find_json_key(&p, "added_tokens") == 0) {
            skip_ws(&p);
            if (*p == '[') {
                p++;
                while (*p && *p != ']') {
                    skip_ws(&p);
                    if (*p == ',') { p++; continue; }
                    if (*p == ']') break;
                    if (*p == '{') {
                        p++;
                        /* Parse added token object: find "id" and "content" */
                        const char *obj_start = p;
                        int at_id = -1;
                        char at_content[1024] = {0};

                        p = obj_start;
                        if (find_json_key(&p, "id") == 0) {
                            at_id = (int)parse_json_int(&p);
                        }

                        p = obj_start;
                        if (find_json_key(&p, "content") == 0) {
                            parse_json_string(&p, at_content, sizeof(at_content));
                        }

                        /* Skip to end of object */
                        p = obj_start;
                        int depth = 1;
                        while (*p && depth > 0) {
                            if (*p == '{') depth++;
                            else if (*p == '}') depth--;
                            else if (*p == '"') {
                                p++;
                                while (*p && *p != '"') {
                                    if (*p == '\\') p++;
                                    if (*p) p++;
                                }
                            }
                            if (*p) p++;
                        }

                        if (at_id >= 0 && at_id < vocab_size && at_content[0]) {
                            free(tok->id_to_text[at_id]);
                            free(tok->id_to_bpe[at_id]);
                            tok->id_to_text[at_id] = strdup(at_content);
                            tok->id_to_bpe[at_id] = strdup(at_content);
                        }
                    } else {
                        skip_json_value(&p);
                    }
                }
            }
        }
    }

    free(json);
    return 0;
}

/* ========================================================================
 * Public API
 * ======================================================================== */

struct smolvlm_tokenizer *smolvlm_tokenizer_load(const char *model_dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/tokenizer.json", model_dir);

    struct smolvlm_tokenizer *tok = (struct smolvlm_tokenizer *)calloc(1, sizeof(*tok));
    if (!tok) return NULL;

    if (load_tokenizer_json(tok, path) != 0) {
        smolvlm_tokenizer_free(tok);
        return NULL;
    }

    return tok;
}

const char *smolvlm_tokenizer_decode(const struct smolvlm_tokenizer *tok, int token_id) {
    if (!tok || token_id < 0 || token_id >= tok->vocab_size) return "";
    return tok->id_to_text[token_id] ? tok->id_to_text[token_id] : "";
}

int *smolvlm_tokenizer_encode(const struct smolvlm_tokenizer *tok, const char *text, int *out_n) {
    if (out_n) *out_n = 0;
    if (!tok || !text || text[0] == '\0') return NULL;

    char *mapped = text_to_bpe_unicode(text);
    if (!mapped) return NULL;

    int *ids = NULL;
    int n_ids = 0;
    if (encode_bpe_word(tok, mapped, &ids, &n_ids) != 0) {
        free(mapped);
        free(ids);
        return NULL;
    }
    free(mapped);

    if (out_n) *out_n = n_ids;
    return ids;
}

void smolvlm_tokenizer_free(struct smolvlm_tokenizer *tok) {
    if (!tok) return;

    if (tok->id_to_text) {
        for (int i = 0; i < tok->vocab_size; i++) free(tok->id_to_text[i]);
        free(tok->id_to_text);
    }
    if (tok->id_to_bpe) {
        for (int i = 0; i < tok->vocab_size; i++) free(tok->id_to_bpe[i]);
        free(tok->id_to_bpe);
    }

    if (tok->merge_map) {
        str_int_entry_t *m = (str_int_entry_t *)tok->merge_map;
        for (int i = 0; i < tok->merge_map_cap; i++) free(m[i].key);
        free(tok->merge_map);
    }
    free(tok->vocab_map);
    free(tok);
}
