/* ============================================================================
 * common.cuh -- Utilidades compartidas: chequeo de errores CUDA y medicion
 *               de tiempo con cudaEvent_t.
 * ==========================================================================*/
#ifndef COMMON_CUH
#define COMMON_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

/* Envuelve cualquier llamada de la API de CUDA que devuelva cudaError_t.
 * Aborta inmediatamente indicando archivo y linea: en un programa de
 * entrenamiento un error silencioso se manifiesta mucho despues como NaN,
 * y es mucho mas caro de diagnosticar. */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "[CUDA] %s:%d '%s' -> %s\n",                       \
                    __FILE__, __LINE__, #call, cudaGetErrorString(_err));      \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* Chequeo despues de un lanzamiento de kernel.
 * Se necesitan DOS comprobaciones distintas:
 *   - cudaGetLastError() detecta errores de lanzamiento (configuracion de
 *     grid/block invalida, demasiada memoria compartida pedida, etc.),
 *   - cudaDeviceSynchronize() detecta errores de ejecucion (acceso ilegal a
 *     memoria), que son asincronos y no aparecen en el lanzamiento.
 * La sincronizacion solo se compila en modo debug (-DVIT_DEBUG) porque
 * serializa el pipeline y falsearia las mediciones de tiempo. */
#ifdef VIT_DEBUG
#define CUDA_CHECK_KERNEL()                                                    \
    do {                                                                       \
        CUDA_CHECK(cudaGetLastError());                                        \
        CUDA_CHECK(cudaDeviceSynchronize());                                   \
    } while (0)
#else
#define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())
#endif

/* Division entera hacia arriba, para calcular el numero de bloques. */
__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

/* Cronometro basado en eventos de CUDA. Mide tiempo de GPU (no de CPU), que es
 * lo correcto para kernels asincronos: cudaEventElapsedTime devuelve el tiempo
 * transcurrido entre dos marcas insertadas en el stream. */
struct CudaTimer {
    cudaEvent_t beg, end;
    CudaTimer()  { CUDA_CHECK(cudaEventCreate(&beg)); CUDA_CHECK(cudaEventCreate(&end)); }
    ~CudaTimer() { cudaEventDestroy(beg); cudaEventDestroy(end); }
    void start() { CUDA_CHECK(cudaEventRecord(beg, 0)); }
    /* Devuelve milisegundos. Incluye la sincronizacion sobre el evento final. */
    float stop() {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventRecord(end, 0));
        CUDA_CHECK(cudaEventSynchronize(end));
        CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
        return ms;
    }
};

/* Memoria de GPU actualmente en uso por el proceso, en MiB. */
inline double gpu_mem_used_mib()
{
    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    return (double)(totalB - freeB) / (1024.0 * 1024.0);
}

/* Tamano de tile para todos los kernels de multiplicacion de matrices.
 * 16x16 = 256 hilos por bloque: multiplo del warp (32), deja suficientes
 * bloques residentes por SM en una T4 (limite de 1024 hilos/bloque y 64 KB de
 * memoria compartida por SM), y cada tile de 16x16 floats ocupa solo 1 KB. */
#define TILE 16

#endif /* COMMON_CUH */
