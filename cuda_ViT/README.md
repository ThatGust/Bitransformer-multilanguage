# Vision Transformer desde cero en CUDA — MNIST

Implementación completa de un Vision Transformer con kernels CUDA propios, sin
cuDNN ni cuBLAS, incluyendo el paso hacia atrás derivado y programado a mano
capa por capa. Entrenado y evaluado sobre MNIST.

## Estructura

```
VisionTransformer/
  data/                 los 4 archivos .ubyte de MNIST (formato IDX)
  src/                  kernels CUDA y el cargador de datos en C
  tests/                verificación de los kernels sin GPU
  scripts/              referencia PyTorch, runner de experimentos, generador de figuras
  results/              CSV producidos por cada corrida
  informe/              informe.tex, figs/, tables/, informe.bib
  colab/                notebook de ejecución en Google Colab
  Makefile
```

### Archivos fuente

| Archivo | Contenido |
|---|---|
| `src/data_loader.c/.h` | Parser IDX big-endian, normalización a [0,1], test con ASCII art |
| `src/common.cuh` | Macros de chequeo de errores CUDA, cronómetro con `cudaEvent_t` |
| `src/gemm.cu` | Multiplicación de matrices tileada: `gemm_nn`, `gemm_nt`, `gemm_tn` |
| `src/attention.cu` | Atención multi-cabeza. **Contiene las dos versiones del kernel crítico** (memoria global vs compartida) |
| `src/layernorm.cu` | LayerNorm, forward y backward analítico |
| `src/patch_embed.cu` | `im2patch`, token [CLS], embeddings posicionales |
| `src/ops.cu` | GELU, residuales, entropía cruzada, AdamW |
| `src/vit.cu` | Ensamblado del modelo: forward y backward completos |
| `src/train.cu` | Bucle de entrenamiento, evaluación, salida CSV |
| `src/bench_attention.cu` | Microbenchmark aislado global vs compartida |

## Uso rápido

```bash
make verify          # verifica los kernels en CPU — no necesita GPU
make                 # compila (por defecto ARCH=sm_75, la T4 de Colab)
make test_data       # comprueba la carga de MNIST

./scripts/run_experiments.sh smoke     # una configuración de prueba, primero
./scripts/run_experiments.sh           # batería completa

python3 scripts/make_report_assets.py  # CSV -> figuras y tablas
tectonic informe/informe.tex           # informe en PDF
```

Para otra GPU: `make ARCH=sm_80` (A100), `sm_60` (P100), `sm_89` (L4).

En Colab, abre `colab/ViT_CUDA_MNIST.ipynb` y sigue las celdas en orden.

## Los dos experimentos

**Experimento 1 — número de parches.** Tres configuraciones: parches de 4x4
(49 parches, 50 tokens), 7x7 (16 parches, 17 tokens) y 14x14 (4 parches, 5
tokens). Se mide precisión, tiempo por época, tiempo de inferencia, memoria de
GPU y número de parámetros.

**Experimento 2 — memoria global vs compartida.** Los dos productos matriciales
de la atención (`QKᵀ` y `P·V`) están implementados dos veces: una leyendo todo
de memoria global y otra con tiling en memoria compartida. `bin/bench_attention`
los mide de forma aislada con `cudaEvent_t`, comprobando primero que ambas
versiones den el mismo resultado.

Los dos experimentos se conectan: el coste de la atención crece como O(T²) con
el número de tokens, así que cuantos más parches, mayor es la matriz de scores y
más rinde el tiling.

## Verificación

`make verify` compila y ejecuta `tests/verify_kernels.cpp`, que:

- emula en CPU **las mismas expresiones de índice** de los kernels tileados
  (misma estructura de bloques, hilos y tiles) y las contrasta contra una
  implementación ingenua, incluyendo tamaños que no son múltiplos de 16 para
  ejercitar las guardas de frontera;
- comprueba que `split_qkv`/`merge_qkv_grad` y `merge_heads`/`split_heads` sean
  inversas exactas, y que `im2patch` cubra la imagen de forma biyectiva;
- contrasta cada gradiente analítico (LayerNorm, softmax con escala, GELU)
  contra su aproximación por diferencias centradas.

No necesita GPU: correrlo antes de compilar ahorra depuración.

`scripts/vit_reference.py` entrena la misma arquitectura en PyTorch (funciona en
CPU, CUDA y MPS) y fija la precisión esperada. No forma parte de la entrega: es
el control que dice si un resultado bajo se debe a los kernels o a los
hiperparámetros.

## Resultados (corrida real en Colab, T4)

| Parche | Tokens | Acc. test | s/época | Inferencia (ms/batch) |
|---|---|---|---|---|
| 4×4 | 50 | 86.95 % | 0.35 | 1.20 |
| 7×7 | 17 | 86.45 % | 0.12 | 0.43 |
| 14×14 | 5 | **92.30 %** | **0.05** | **0.19** |

El parche de 14×14 gana con claridad en precisión y es a la vez el más barato.
Entre 4×4 y 7×7 la diferencia (0.5 pp) es indistinguible del ruido de una sola
corrida — ver la discusión completa en `informe/informe.pdf`.

Microbenchmark aislado (memoria global vs. compartida): el kernel `QK^T` gana
hasta 2.08× con tiling, pero `P·V` resulta **más lento** con tiling en las tres
configuraciones (0.62×–0.69×). No es un error — ambas versiones dan resultados
bit a bit idénticos — sino un efecto real de cuántas iteraciones de tiling
necesita cada kernel; el informe explica el mecanismo en detalle.

## Configuración por defecto

| | |
|---|---|
| Embedding | D = 64, 4 cabezas (D_h = 16) |
| Bloques | 2, pre-norm |
| MLP | oculta 128, activación GELU |
| Cabeza | token [CLS] → LayerNorm → lineal a 10 clases |
| Posicionales | aprendibles |
| Optimizador | AdamW, lr 1e-3, weight decay 0.01 |
| Datos | 5 000 train / 2 000 test, batch 64, 15 épocas |

Todo es configurable por línea de comandos (`./bin/vit_train --help`). El
subconjunto de datos se cambia con `--train-n` y `--test-n`; para el dataset
completo, `--train-n 60000 --test-n 10000`.

## Presupuesto de tiempo

`vit_train` imprime el tiempo de la primera época en cuanto termina. Si supera
el límite (`--max-epoch-sec`, 30 s por defecto), reduce el subconjunto a la
mitad y reinicia desde cero, informando de qué cambió y por qué. Se desactiva
con `--no-auto-shrink`.
