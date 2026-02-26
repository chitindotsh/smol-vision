# qwen_asr — Qwen3-ASR Pure C Inference Engine
# smolvlm  — SmolVLM-Instruct Pure C Inference Engine
# Makefile

CC = gcc
CFLAGS_BASE = -Wall -Wextra -O3 -march=native -ffast-math
LDFLAGS = -lm -lpthread

# Platform detection
UNAME_S := $(shell uname -s)

# Source files — ASR
SRCS = qwen_asr.c qwen_asr_kernels.c qwen_asr_kernels_generic.c qwen_asr_kernels_neon.c qwen_asr_kernels_avx.c qwen_asr_audio.c qwen_asr_encoder.c qwen_asr_decoder.c qwen_asr_tokenizer.c qwen_asr_safetensors.c
OBJS = $(SRCS:.c=.o)
MAIN = main.c
TARGET = qwen_asr

# Source files — SmolVLM
SMOLVLM_SRCS = smolvlm.c smolvlm_vision.c smolvlm_decoder.c smolvlm_image.c smolvlm_tokenizer.c \
               qwen_asr_kernels.c qwen_asr_kernels_generic.c qwen_asr_kernels_neon.c \
               qwen_asr_kernels_avx.c qwen_asr_safetensors.c
SMOLVLM_OBJS = $(SMOLVLM_SRCS:.c=.o)
SMOLVLM_MAIN = smolvlm_main.c
SMOLVLM_TARGET = smolvlm

# Debug build flags
DEBUG_CFLAGS = -Wall -Wextra -g -O0 -DDEBUG -fsanitize=address

.PHONY: all clean clean-smolvlm debug info help blas blas-smolvlm debug-smolvlm test test-stream-cache

# Default: show available targets
all: help

help:
	@echo "qwen_asr / smolvlm — Pure C Inference - Build Targets"
	@echo ""
	@echo "ASR targets:"
	@echo "  make blas            - Qwen3-ASR with BLAS (Accelerate/OpenBLAS)"
	@echo "  make debug           - Qwen3-ASR debug build with AddressSanitizer"
	@echo ""
	@echo "SmolVLM targets:"
	@echo "  make blas-smolvlm    - SmolVLM-Instruct with BLAS"
	@echo "  make debug-smolvlm   - SmolVLM-Instruct debug build"
	@echo ""
	@echo "Other:"
	@echo "  make test            - Run ASR regression suite"
	@echo "  make clean           - Remove all build artifacts"
	@echo "  make info            - Show build configuration"
	@echo ""
	@echo "ASR example:    make blas && ./qwen_asr -d model_dir -i audio.wav"
	@echo "SmolVLM example: make blas-smolvlm && ./smolvlm -d smolvlm-instruct -i image.pnm -p 'Describe'"

# =============================================================================
# Backend: blas — ASR (Accelerate on macOS, OpenBLAS on Linux)
# =============================================================================
ifeq ($(UNAME_S),Darwin)
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK
blas: LDFLAGS += -framework Accelerate
else
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
blas: LDFLAGS += -lopenblas
endif
blas:
	@$(MAKE) clean
	@$(MAKE) $(TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"
	@echo ""
	@echo "Built qwen_asr with BLAS backend"

# =============================================================================
# Backend: blas — SmolVLM
# =============================================================================
ifeq ($(UNAME_S),Darwin)
blas-smolvlm: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK
blas-smolvlm: LDFLAGS += -framework Accelerate
else
blas-smolvlm: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
blas-smolvlm: LDFLAGS += -lopenblas
endif
blas-smolvlm:
	@$(MAKE) clean-smolvlm
	@$(MAKE) $(SMOLVLM_TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"
	@echo ""
	@echo "Built smolvlm with BLAS backend"

# =============================================================================
# Build rules — ASR
# =============================================================================
$(TARGET): $(OBJS) main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Build rules — SmolVLM
$(SMOLVLM_TARGET): $(SMOLVLM_OBJS) smolvlm_main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Debug builds
debug: CFLAGS = $(DEBUG_CFLAGS)
debug: LDFLAGS += -fsanitize=address
debug:
	@$(MAKE) clean
	@$(MAKE) $(TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"

debug-smolvlm: CFLAGS = $(DEBUG_CFLAGS)
debug-smolvlm: LDFLAGS += -fsanitize=address
debug-smolvlm:
	@$(MAKE) clean-smolvlm
	@$(MAKE) $(SMOLVLM_TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"

# =============================================================================
# Utilities
# =============================================================================
clean:
	rm -f $(OBJS) $(SMOLVLM_OBJS) main.o smolvlm_main.o $(TARGET) $(SMOLVLM_TARGET)

clean-smolvlm:
	rm -f $(SMOLVLM_OBJS) smolvlm_main.o $(SMOLVLM_TARGET)

info:
	@echo "Platform: $(UNAME_S)"
	@echo "Compiler: $(CC)"
	@echo ""
ifeq ($(UNAME_S),Darwin)
	@echo "Backend: blas (Apple Accelerate)"
else
	@echo "Backend: blas (OpenBLAS)"
endif

test:
	./asr_regression.py --binary ./qwen_asr --model-dir qwen3-asr-1.7b

# =============================================================================
# Dependencies — ASR
# =============================================================================
qwen_asr.o: qwen_asr.c qwen_asr.h qwen_asr_kernels.h qwen_asr_safetensors.h qwen_asr_audio.h qwen_asr_tokenizer.h
qwen_asr_kernels.o: qwen_asr_kernels.c qwen_asr_kernels.h qwen_asr_kernels_impl.h
qwen_asr_kernels_generic.o: qwen_asr_kernels_generic.c qwen_asr_kernels_impl.h
qwen_asr_kernels_neon.o: qwen_asr_kernels_neon.c qwen_asr_kernels_impl.h
qwen_asr_kernels_avx.o: qwen_asr_kernels_avx.c qwen_asr_kernels_impl.h
qwen_asr_audio.o: qwen_asr_audio.c qwen_asr_audio.h
qwen_asr_encoder.o: qwen_asr_encoder.c qwen_asr.h qwen_asr_kernels.h qwen_asr_safetensors.h
qwen_asr_decoder.o: qwen_asr_decoder.c qwen_asr.h qwen_asr_kernels.h qwen_asr_safetensors.h
qwen_asr_tokenizer.o: qwen_asr_tokenizer.c qwen_asr_tokenizer.h
qwen_asr_safetensors.o: qwen_asr_safetensors.c qwen_asr_safetensors.h
main.o: main.c qwen_asr.h qwen_asr_kernels.h

# =============================================================================
# Dependencies — SmolVLM
# =============================================================================
smolvlm.o: smolvlm.c smolvlm.h smolvlm_tokenizer.h qwen_asr_kernels.h qwen_asr_safetensors.h
smolvlm_vision.o: smolvlm_vision.c smolvlm.h qwen_asr_kernels.h qwen_asr_safetensors.h
smolvlm_decoder.o: smolvlm_decoder.c smolvlm.h qwen_asr_kernels.h qwen_asr_safetensors.h
smolvlm_image.o: smolvlm_image.c smolvlm.h
smolvlm_tokenizer.o: smolvlm_tokenizer.c smolvlm_tokenizer.h
smolvlm_main.o: smolvlm_main.c smolvlm.h qwen_asr_kernels.h
