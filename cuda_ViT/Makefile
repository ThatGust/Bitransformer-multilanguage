# ============================================================================
# Makefile del ViT en CUDA.
#
#   make            -> compila vit_train, bench_attention y test_loader
#   make verify     -> compila y corre los chequeos de kernels en CPU (sin GPU)
#   make test_data  -> comprueba la carga de MNIST (ASCII art)
#   make clean
#
# ARCH: sm_75 es la arquitectura Turing de la T4 de Colab. Para otra GPU:
#   make ARCH=sm_80        (A100)
#   make ARCH=sm_60        (P100, tambien frecuente en Colab)
# ============================================================================

ARCH    ?= sm_75
NVCC    ?= nvcc
CC      ?= gcc
BIN     ?= bin

# -O3 optimizacion completa; --use_fast_math habilita las versiones rapidas de
# las funciones trascendentes (suficiente precision para entrenar en fp32);
# -lineinfo conserva la correspondencia con las lineas fuente para nsys/ncu.
NVCCFLAGS = -O3 -arch=$(ARCH) -std=c++14 -lineinfo -Xcompiler -Wall
CFLAGS    = -O2 -Wall -Wextra

# Añadir -DVIT_DEBUG activa cudaDeviceSynchronize tras cada kernel (mas lento,
# pero localiza el kernel exacto que falla).
# NVCCFLAGS += -DVIT_DEBUG

SRC       = src
CU_COMMON = $(SRC)/gemm.cu $(SRC)/attention.cu $(SRC)/layernorm.cu \
            $(SRC)/patch_embed.cu $(SRC)/ops.cu $(SRC)/vit.cu

.PHONY: all clean verify test_data dirs

all: dirs $(BIN)/vit_train $(BIN)/bench_attention $(BIN)/test_loader

dirs:
	@mkdir -p $(BIN) results informe/figs

# El loader es C puro y se compila con el compilador de C: su cabecera usa
# extern "C", asi que el enlazado con las unidades CUDA (C++) es correcto.
$(BIN)/data_loader.o: $(SRC)/data_loader.c $(SRC)/data_loader.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN)/vit_train: $(SRC)/train.cu $(CU_COMMON) $(BIN)/data_loader.o
	$(NVCC) $(NVCCFLAGS) $(SRC)/train.cu $(CU_COMMON) $(BIN)/data_loader.o -o $@

$(BIN)/bench_attention: $(SRC)/bench_attention.cu $(SRC)/attention.cu
	$(NVCC) $(NVCCFLAGS) $(SRC)/bench_attention.cu $(SRC)/attention.cu -o $@

$(BIN)/test_loader: $(SRC)/data_loader.c $(SRC)/data_loader.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -DDATA_LOADER_TEST $< -o $@

# Chequeo de la indexacion de los kernels y de los gradientes analiticos.
# Emula en CPU las mismas expresiones de indice que los kernels: NO necesita GPU,
# se puede correr en cualquier maquina antes de subir nada a Colab.
$(BIN)/verify_kernels: tests/verify_kernels.cpp
	@mkdir -p $(BIN)
	c++ -O2 -std=c++17 $< -o $@

verify: $(BIN)/verify_kernels
	./$(BIN)/verify_kernels

test_data: $(BIN)/test_loader
	./$(BIN)/test_loader data

clean:
	rm -rf $(BIN)
