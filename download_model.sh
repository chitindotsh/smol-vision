#!/bin/bash
# Download model files from HuggingFace.
#
# Usage:
#   ./download_model.sh
#   ./download_model.sh --model smolvlm
#   ./download_model.sh --model smolvlm --dir my-model-dir
#
# Options:
#   --model MODEL   Choose model variant (see list below)
#   --dir DIR       Override output directory

set -e

MODEL_CHOICE=""
MODEL_DIR=""

usage() {
    echo "Usage: $0 [--model MODEL] [--dir DIR]"
    echo ""
    echo "Available models:"
    echo "  smolvlm   SmolVLM-Instruct (2.2B vision-language model)"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL_CHOICE="$2"
            shift 2
            ;;
        --dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

choose_model_interactive() {
    echo "Select model to download:"
    echo "  1) smolvlm (SmolVLM-Instruct 2.2B)"
    echo ""
    while true; do
        read -r -p "Enter choice [1]: " ans
        case "$ans" in
            1|smolvlm|SmolVLM|SMOLVLM|"")
                MODEL_CHOICE="smolvlm"
                return
                ;;
            *)
                echo "Please choose 1 (smolvlm)."
                ;;
        esac
    done
}

if [[ -z "$MODEL_CHOICE" ]]; then
    choose_model_interactive
fi

case "$MODEL_CHOICE" in
    smolvlm|SmolVLM)
        MODEL_ID="HuggingFaceTB/SmolVLM-Instruct"
        if [[ -z "$MODEL_DIR" ]]; then MODEL_DIR="smolvlm-instruct"; fi
        FILES=(
            "config.json"
            "generation_config.json"
            "preprocessor_config.json"
            "tokenizer.json"
            "tokenizer_config.json"
            "special_tokens_map.json"
            "model.safetensors.index.json"
            "model-00001-of-00002.safetensors"
            "model-00002-of-00002.safetensors"
        )
        ;;
    *)
        echo "Invalid --model value: $MODEL_CHOICE"
        usage
        exit 1
        ;;
esac

echo "Downloading ${MODEL_ID} to ${MODEL_DIR}/"
echo ""

mkdir -p "${MODEL_DIR}"

BASE_URL="https://huggingface.co/${MODEL_ID}/resolve/main"

for file in "${FILES[@]}"; do
    dest="${MODEL_DIR}/${file}"
    if [[ -f "${dest}" ]]; then
        echo "  [skip] ${file} (already exists)"
    else
        echo "  [download] ${file}..."
        curl -fL -o "${dest}" "${BASE_URL}/${file}" --progress-bar
        echo "  [done] ${file}"
    fi
done

echo ""
echo "Download complete. Files in ${MODEL_DIR}/"
ls -lh "${MODEL_DIR}/"
