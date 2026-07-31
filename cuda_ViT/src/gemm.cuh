/* ============================================================================
 * gemm.cuh -- Multiplicacion de matrices densa (no batcheada), usada por todas
 *             las capas lineales del ViT: proyeccion de parches, QKV, la
 *             proyeccion de salida de la atencion, el MLP y la cabeza de
 *             clasificacion.
 *
 * Convencion de nombres (estilo BLAS): la letra indica si el operando se usa
 * transpuesto (N = normal, T = transpuesto).
 *
 *   gemm_nn : C[M,N] = A[M,K] * B[K,N]        (+ bias opcional por columna)
 *   gemm_nt : C[M,N] = A[M,K] * B[N,K]^T
 *   gemm_tn : C[M,N] = A[K,M]^T * B[K,N]
 *
 * Los tres casos aparecen naturalmente en el backward de una capa lineal
 * Y = X*W + b, con X[M,K], W[K,N]:
 *   dX[M,K] = dY[M,N] * W[K,N]^T   -> gemm_nt
 *   dW[K,N] = X[M,K]^T * dY[M,N]   -> gemm_tn
 *   db[N]   = sum_m dY[m,N]        -> bias_backward
 * ==========================================================================*/
#ifndef GEMM_CUH
#define GEMM_CUH

/* bias puede ser NULL si la capa no tiene sesgo. */
void gemm_nn(const float *A, const float *B, const float *bias, float *C,
             int M, int N, int K, cudaStream_t stream = 0);
void gemm_nt(const float *A, const float *B, float *C,
             int M, int N, int K, cudaStream_t stream = 0);
void gemm_tn(const float *A, const float *B, float *C,
             int M, int N, int K, cudaStream_t stream = 0);

/* db[n] = sum_m dY[m,n]. Reduccion por columnas. */
void bias_backward(const float *dY, float *db, int M, int N, cudaStream_t stream = 0);

#endif /* GEMM_CUH */
