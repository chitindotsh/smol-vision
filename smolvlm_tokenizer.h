/*
 * smolvlm_tokenizer.h - GPT-2 BPE tokenizer for SmolVLM
 *
 * Loads vocab + merges from tokenizer.json.
 */

#ifndef SMOLVLM_TOKENIZER_H
#define SMOLVLM_TOKENIZER_H

struct smolvlm_tokenizer {
    char **id_to_text;   /* [vocab_size] decoded text strings */
    char **id_to_bpe;    /* [vocab_size] raw BPE token strings */
    int vocab_size;

    /* Internal hash maps */
    void *vocab_map;
    int vocab_map_cap;
    void *merge_map;
    int merge_map_cap;
};

/* Load tokenizer from model_dir/tokenizer.json */
struct smolvlm_tokenizer *smolvlm_tokenizer_load(const char *model_dir);

/* Decode a single token ID to text. Returns pointer to internal string. */
const char *smolvlm_tokenizer_decode(const struct smolvlm_tokenizer *tok, int token_id);

/* Encode UTF-8 text into token IDs using BPE.
 * Returns malloc'd array of token IDs and sets *out_n.
 * Returns NULL on error. */
int *smolvlm_tokenizer_encode(const struct smolvlm_tokenizer *tok, const char *text, int *out_n);

/* Free tokenizer */
void smolvlm_tokenizer_free(struct smolvlm_tokenizer *tok);

#endif /* SMOLVLM_TOKENIZER_H */
