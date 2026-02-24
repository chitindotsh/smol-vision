#!/bin/bash
# Download Qwen3-ASR model files from HuggingFace.
#
# Usage:
#   ./download_model.sh
#   ./download_model.sh --model small
#   ./download_model.sh --model large --dir my-model-dir
#
# Options:
#   --model small|large   Choose 0.6B (small) or 1.7B (large)
#   --dir DIR             Override output directory

set -e

MODEL_CHOICE=""
MODEL_DIR=""

usage() {
    echo "Usage: $0 [--model small|large] [--dir DIR]"
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
    echo "Select model size to download:"
    echo "  1) small (Qwen3-ASR-0.6B)"
    echo "  2) large (Qwen3-ASR-1.7B)"
    echo "  3) omni (Qwen2.5-Omni-7B)"
    echo "  4) 30B (Qwen3-Omni-30B-A3B-Instruct)"
    echo ""
    while true; do
        read -r -p "Enter choice [1/2]: " ans
        case "$ans" in
            1|small|Small|SMALL)
                MODEL_CHOICE="small"
                return
                ;;
            2|large|Large|LARGE)
                MODEL_CHOICE="large"
                return
                ;;
            3|omni|Omni|OMNI)
                MODEL_CHOICE="omni"
                return
                ;;
            4|30b|30B|30b|30B)
                MODEL_CHOICE="30b"
                return
                ;;
            *)
                echo "Please choose 1 (small), 2 (large), 3 (omni), or 4 (30B)."
                ;;
        esac
    done
}

if [[ -z "$MODEL_CHOICE" ]]; then
    choose_model_interactive
fi

case "$MODEL_CHOICE" in
    small|0.6b|0.6B)
        MODEL_ID="Qwen/Qwen3-ASR-0.6B"
        if [[ -z "$MODEL_DIR" ]]; then MODEL_DIR="qwen3-asr-0.6b"; fi
        FILES=(
            "config.json"
            "generation_config.json"
            "model.safetensors"
            "vocab.json"
            "merges.txt"
        )
        ;;
    large|1.7b|1.7B)
        MODEL_ID="Qwen/Qwen3-ASR-1.7B"
        if [[ -z "$MODEL_DIR" ]]; then MODEL_DIR="qwen3-asr-1.7b"; fi
        FILES=(
            "config.json"
            "generation_config.json"
            "model.safetensors.index.json"
            "model-00001-of-00002.safetensors"
            "model-00002-of-00002.safetensors"
            "vocab.json"
            "merges.txt"
        )
        ;;
    omni|Omni|OMNI)
        MODEL_ID="Qwen/Qwen2.5-Omni-7B"
        if [[ -z "$MODEL_DIR" ]]; then MODEL_DIR="qwen2.5-omni-7b"; fi
        FILES=(
            "config.json"
            "generation_config.json"
            "model.safetensors.index.json"
            "model-00001-of-00005.safetensors"
            "model-00002-of-00005.safetensors"
            "model-00003-of-00005.safetensors"
            "model-00004-of-00005.safetensors"
            "model-00005-of-00005.safetensors"
            "vocab.json"
            "merges.txt"
        )
        ;;
    30b)
        MODEL_ID="Qwen/Qwen3-Omni-30B-A3B-Instruct"
        if [[ -z "$MODEL_DIR" ]]; then MODEL_DIR="qwen3-omni-30b"; fi
        FILES=(
            "config.json"
            "generation_config.json"
            "model.safetensors.index.json"
            "model-00001-of-00015.safetensors"
            "model-00002-of-00015.safetensors"
            "model-00003-of-00015.safetensors"
            "model-00004-of-00015.safetensors"
            "model-00005-of-00015.safetensors"
            "model-00006-of-00015.safetensors"
            "model-00007-of-00015.safetensors"
            "model-00008-of-00015.safetensors"
            "model-00009-of-00015.safetensors"
            "model-00010-of-00015.safetensors"
            "model-00011-of-00015.safetensors"
            "model-00012-of-00015.safetensors"
            "model-00013-of-00015.safetensors"
            "model-00014-of-00015.safetensors"
            "model-00015-of-00015.safetensors"
            "vocab.json"
            "merges.txt"
        )
        ;;
    *)
        echo "Invalid --model value: $MODEL_CHOICE"
        echo "Use: --model small or --model large"
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
