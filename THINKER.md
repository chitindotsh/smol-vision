# Thinker Mode

Thinker mode uses the Qwen3-Omni-30B MoE decoder for free-form text generation
instead of ASR transcription. It accepts text input, audio input, or both.

## Quick Start

```bash
# Text-only chat
./qwen_asr -d qwen3-omni-30b --thinker --text "What is 2+2?" \
  --prompt "You are a helpful assistant" --moe-preload

# Audio question-answering
./qwen_asr -d qwen3-omni-30b --thinker -i question.wav \
  --prompt "Answer the user's question" --moe-preload

# Audio + text combined
./qwen_asr -d qwen3-omni-30b --thinker -i meeting.wav \
  --text "Summarize this audio" --moe-preload
```

## Sampling Parameters

Thinker mode uses temperature sampling by default to produce diverse,
non-repetitive text. These parameters only affect thinker mode — all ASR
paths remain greedy.

| Flag | Default | Description |
|------|---------|-------------|
| `--temperature <f>` | 0.7 | Controls randomness. Higher = more creative, lower = more focused. 0 = greedy (deterministic). |
| `--temp <f>` | — | Alias for `--temperature` |
| `--repeat-penalty <f>` | 1.1 | Penalizes recently generated tokens. 1.0 = off. Higher values reduce repetition. |
| `--top-k <n>` | 40 | Only sample from the top-k most likely tokens. 0 = off (consider all tokens). |
| `--max-tokens <n>` | 2048 | Maximum number of tokens to generate. |

## Examples

### Greedy decoding (deterministic, like previous behavior)

```bash
./qwen_asr -d qwen3-omni-30b --thinker --text "What is the capital of France?" \
  --prompt "You are a helpful assistant" --temp 0 --moe-preload
```

### Creative writing (high temperature)

```bash
./qwen_asr -d qwen3-omni-30b --thinker \
  --text "Write a short poem about the ocean" \
  --prompt "You are a creative writer" \
  --temp 1.0 --top-k 50 --max-tokens 512 --moe-preload
```

### Precise answers (low temperature, strong repetition penalty)

```bash
./qwen_asr -d qwen3-omni-30b --thinker \
  --text "List the first 10 prime numbers" \
  --prompt "You are a math tutor. Be concise." \
  --temp 0.3 --repeat-penalty 1.2 --moe-preload
```

### Silent mode (only output the generated text, no status)

```bash
./qwen_asr -d qwen3-omni-30b --thinker --silent \
  --text "Translate to French: Good morning" \
  --prompt "You are a translator" --moe-preload
```

### Piping output

```bash
ANSWER=$(./qwen_asr -d qwen3-omni-30b --thinker --silent \
  --text "What is 2+2?" --prompt "Answer with just the number" --moe-preload)
echo "The answer is: $ANSWER"
```

## Notes

- `--moe-preload` is recommended for the 30B MoE model. It pre-faults all
  expert weight pages into RAM, avoiding page-fault stalls during generation.
  This requires ~60 GB of available memory.
- The system prompt (`--prompt`) sets the assistant's behavior and is placed
  in the `<|im_start|>system` turn of the chat template.
- Generation stops at `<|im_end|>` or `<|endoftext|>` tokens, or when
  `--max-tokens` is reached.
