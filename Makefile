# smol-vision — SmolVLM-Instruct Pure C Inference Engine
# Makefile

CC = gcc
CFLAGS_BASE = -Wall -Wextra -O3 -march=native -ffast-math
LDFLAGS = -lm -lpthread

# Platform detection
UNAME_S := $(shell uname -s)

# Source files
SRCS = smolvlm.c smolvlm_vision.c smolvlm_decoder.c smolvlm_image.c smolvlm_tokenizer.c \
       common_kernels.c common_kernels_generic.c common_kernels_neon.c \
       common_kernels_avx.c common_safetensors.c
OBJS = $(SRCS:.c=.o)
MAIN = smolvlm_main.c
TARGET = smolvlm

# Debug build flags
DEBUG_CFLAGS = -Wall -Wextra -g -O0 -DDEBUG -fsanitize=address

.PHONY: all clean debug info help blas test-images

# Default: show available targets
all: help

help:
	@echo "smol-vision — SmolVLM-Instruct Pure C Inference Engine"
	@echo ""
	@echo "Build targets:"
	@echo "  make blas          - Build with BLAS (Accelerate/OpenBLAS)"
	@echo "  make debug         - Debug build with AddressSanitizer"
	@echo ""
	@echo "Test targets:"
	@echo "  make test-images   - Run image loading test suite"
	@echo ""
	@echo "Other:"
	@echo "  make clean         - Remove all build artifacts"
	@echo "  make info          - Show build configuration"
	@echo ""
	@echo "Example: make blas && ./smolvlm -d smolvlm-instruct -i image.jpg -p 'Describe'"

# =============================================================================
# Backend: blas (Accelerate on macOS, OpenBLAS on Linux)
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
	@echo "Built smolvlm with BLAS backend"

# =============================================================================
# Build rules
# =============================================================================
$(TARGET): $(OBJS) smolvlm_main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Debug build
debug: CFLAGS = $(DEBUG_CFLAGS)
debug: LDFLAGS += -fsanitize=address
debug:
	@$(MAKE) clean
	@$(MAKE) $(TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"

# =============================================================================
# Test
# =============================================================================
TEST_IMG_BIN = test_smolvlm_images
test-images: smolvlm_image.o test_smolvlm_images.o
	$(CC) $(CFLAGS_BASE) -o $(TEST_IMG_BIN) $^ -lm
	./$(TEST_IMG_BIN) test_images

test_smolvlm_images.o: test_smolvlm_images.c smolvlm.h stb_image.h

# =============================================================================
# Utilities
# =============================================================================
clean:
	rm -f $(OBJS) smolvlm_main.o test_smolvlm_images.o $(TARGET) $(TEST_IMG_BIN)

info:
	@echo "Platform: $(UNAME_S)"
	@echo "Compiler: $(CC)"
	@echo ""
ifeq ($(UNAME_S),Darwin)
	@echo "Backend: blas (Apple Accelerate)"
else
	@echo "Backend: blas (OpenBLAS)"
endif

# =============================================================================
# Dependencies
# =============================================================================
smolvlm.o: smolvlm.c smolvlm.h smolvlm_tokenizer.h common_kernels.h common_safetensors.h
smolvlm_vision.o: smolvlm_vision.c smolvlm.h common_kernels.h common_safetensors.h
smolvlm_decoder.o: smolvlm_decoder.c smolvlm.h common_kernels.h common_safetensors.h
smolvlm_image.o: smolvlm_image.c smolvlm.h stb_image.h
smolvlm_tokenizer.o: smolvlm_tokenizer.c smolvlm_tokenizer.h
smolvlm_main.o: smolvlm_main.c smolvlm.h common_kernels.h
common_kernels.o: common_kernels.c common_kernels.h common_kernels_impl.h
common_kernels_generic.o: common_kernels_generic.c common_kernels_impl.h
common_kernels_neon.o: common_kernels_neon.c common_kernels_impl.h
common_kernels_avx.o: common_kernels_avx.c common_kernels_impl.h
common_safetensors.o: common_safetensors.c common_safetensors.h
