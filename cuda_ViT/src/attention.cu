/* ============================================================================
 * attention.cu -- Kernels de Multi-Head Self-Attention.
 *
 * -------------------------- EXPERIMENTO 2 ---------------------------------
 * Cada producto matricial por cabeza esta implementado dos veces:
 *
 *   Version A (*_global_kernel): cada hilo calcula un elemento de la salida y
 *   lee sus K pares de operandos DIRECTAMENTE de memoria global. El elemento
 *   A[m,k] lo vuelven a leer los N hilos de la fila m, y B[k,n] los M hilos de
 *   la columna n. El trafico total contra memoria global es O(M*N*K).
 *
 *   Version B (*_shared_kernel): bloques de 16x16 que cooperan para copiar
 *   tiles de 16x16 a memoria compartida; cada valor traido se reutiliza 16
 *   veces dentro del bloque. El trafico baja a O(M*N*K/TILE).
 *
 * Ambas hacen exactamente la misma aritmetica y producen el mismo resultado
 * (salvo el orden de acumulacion en punto flotante), asi que la diferencia de
 * tiempo medida es atribuible solo a la jerarquia de memoria.
 *
 * Por que estos kernels y no el patch embedding: el coste de la atencion crece
 * como O(T^2 * Dh) con el numero de tokens T, y T lo fija el tamano de parche
 * (T = 50, 17 y 5 para parches de 4x4, 7x7 y 14x14). Esto conecta el
 * experimento 1 con el 2: cuantos mas parches, mayor la matriz de scores y mas
 * se nota el ahorro de trafico del tiling.
 * ==========================================================================*/
#include "common.cuh"
#include "attention.cuh"

#include <cmath>

/* ==========================================================================
 *  C[M,N] = alpha * A[M,K] * B[N,K]^T        (QK^T)
 * ========================================================================*/

/* --- Version A: solo memoria global --- */
__global__ void bmm_nt_global_kernel(const float *__restrict__ A,
                                     const float *__restrict__ B,
                                     float *__restrict__ C,
                                     int M, int N, int K,
                                     int sA, int sB, int sC, float alpha)
{
    const int bh  = blockIdx.z;                        /* indice de (batch,cabeza) */
    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    if (row >= M || col >= N) return;

    const float *a = A + (size_t)bh * sA + (size_t)row * K;
    const float *b = B + (size_t)bh * sB + (size_t)col * K;

    float acc = 0.0f;
    for (int i = 0; i < K; ++i) acc += a[i] * b[i];   /* K lecturas globales por hilo */

    C[(size_t)bh * sC + (size_t)row * N + col] = alpha * acc;
}

/* --- Version B: tiling en memoria compartida --- */
__global__ void bmm_nt_shared_kernel(const float *__restrict__ A,
                                     const float *__restrict__ B,
                                     float *__restrict__ C,
                                     int M, int N, int K,
                                     int sA, int sB, int sC, float alpha)
{
    /* 2 tiles * 16 * 17 floats = 2.1 KB por bloque: permite muchos bloques
     * residentes por SM (la T4 tiene 64 KB de shared por SM). El padding +1
     * evita conflictos de banco al leer columnas. */
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    const int bh  = blockIdx.z;
    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    const float *a = A + (size_t)bh * sA;
    const float *b = B + (size_t)bh * sB;
    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        int ak = t + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? a[(size_t)row * K + ak] : 0.0f;

        /* Bs[n_local][i] = B[bloque_n + n_local, t+i]: threadIdx.x recorre K
         * para que los hilos de un warp lean posiciones contiguas de B. */
        int brow = blockIdx.x * TILE + threadIdx.y;
        int bk   = t + threadIdx.x;
        Bs[threadIdx.y][threadIdx.x] = (brow < N && bk < K) ? b[(size_t)brow * K + bk] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int i = 0; i < TILE; ++i)
            acc += As[threadIdx.y][i] * Bs[threadIdx.x][i];
        __syncthreads();
    }

    if (row < M && col < N)
        C[(size_t)bh * sC + (size_t)row * N + col] = alpha * acc;
}

/* ==========================================================================
 *  C[M,N] = alpha * A[M,K] * B[K,N]          (P*V)
 * ========================================================================*/

__global__ void bmm_nn_global_kernel(const float *__restrict__ A,
                                     const float *__restrict__ B,
                                     float *__restrict__ C,
                                     int M, int N, int K,
                                     int sA, int sB, int sC, float alpha)
{
    const int bh  = blockIdx.z;
    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    if (row >= M || col >= N) return;

    const float *a = A + (size_t)bh * sA + (size_t)row * K;
    const float *b = B + (size_t)bh * sB;

    float acc = 0.0f;
    /* El acceso a b tiene stride N entre iteraciones: dentro de un warp los
     * hilos (col consecutivo) si leen contiguo, pero cada iteracion del bucle
     * salta a otra linea de cache. Es exactamente el patron que el tiling
     * arregla. */
    for (int i = 0; i < K; ++i) acc += a[i] * b[(size_t)i * N + col];

    C[(size_t)bh * sC + (size_t)row * N + col] = alpha * acc;
}

__global__ void bmm_nn_shared_kernel(const float *__restrict__ A,
                                     const float *__restrict__ B,
                                     float *__restrict__ C,
                                     int M, int N, int K,
                                     int sA, int sB, int sC, float alpha)
{
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    const int bh  = blockIdx.z;
    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    const float *a = A + (size_t)bh * sA;
    const float *b = B + (size_t)bh * sB;
    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        int ak = t + threadIdx.x;
        int bk = t + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? a[(size_t)row * K + ak] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? b[(size_t)bk * N + col] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int i = 0; i < TILE; ++i)
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N)
        C[(size_t)bh * sC + (size_t)row * N + col] = alpha * acc;
}

/* ==========================================================================
 *  C[M,N] = alpha * A[K,M]^T * B[K,N]        (dV y dK del backward)
 * ========================================================================*/

__global__ void bmm_tn_global_kernel(const float *__restrict__ A,
                                     const float *__restrict__ B,
                                     float *__restrict__ C,
                                     int M, int N, int K,
                                     int sA, int sB, int sC, float alpha)
{
    const int bh  = blockIdx.z;
    const int row = blockIdx.y * TILE + threadIdx.y;   /* sobre M */
    const int col = blockIdx.x * TILE + threadIdx.x;   /* sobre N */
    if (row >= M || col >= N) return;

    const float *a = A + (size_t)bh * sA;
    const float *b = B + (size_t)bh * sB;

    float acc = 0.0f;
    for (int i = 0; i < K; ++i) acc += a[(size_t)i * M + row] * b[(size_t)i * N + col];

    C[(size_t)bh * sC + (size_t)row * N + col] = alpha * acc;
}

__global__ void bmm_tn_shared_kernel(const float *__restrict__ A,
                                     const float *__restrict__ B,
                                     float *__restrict__ C,
                                     int M, int N, int K,
                                     int sA, int sB, int sC, float alpha)
{
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    const int bh  = blockIdx.z;
    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    const float *a = A + (size_t)bh * sA;
    const float *b = B + (size_t)bh * sB;
    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        /* As[i][m_local] = A[t+i, bloque_m + m_local] */
        int ak = t + threadIdx.y;
        int am = blockIdx.y * TILE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (ak < K && am < M) ? a[(size_t)ak * M + am] : 0.0f;

        int bk = t + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? b[(size_t)bk * N + col] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int i = 0; i < TILE; ++i)
            acc += As[i][threadIdx.y] * Bs[i][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N)
        C[(size_t)bh * sC + (size_t)row * N + col] = alpha * acc;
}

/* ==========================================================================
 *  Softmax por filas
 *  Un bloque por fila. Se usa el truco estandar de restar el maximo antes de
 *  exponenciar: sin el, exp() de scores grandes desborda a inf y el resultado
 *  es NaN. Dos reducciones en memoria compartida (maximo y suma).
 * ========================================================================*/
#define SOFTMAX_THREADS 128

__global__ void softmax_rows_kernel(const float *__restrict__ x,
                                    float *__restrict__ y,
                                    int n_cols)
{
    __shared__ float red[SOFTMAX_THREADS];
    const size_t base = (size_t)blockIdx.x * n_cols;
    const int tid = threadIdx.x;

    float m = -INFINITY;
    for (int j = tid; j < n_cols; j += blockDim.x) m = fmaxf(m, x[base + j]);
    red[tid] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    const float row_max = red[0];
    __syncthreads();

    float sum = 0.0f;
    for (int j = tid; j < n_cols; j += blockDim.x) {
        float e = __expf(x[base + j] - row_max);
        y[base + j] = e;                     /* se guarda sin normalizar */
        sum += e;
    }
    red[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float inv = 1.0f / red[0];

    for (int j = tid; j < n_cols; j += blockDim.x) y[base + j] *= inv;
}

/* Backward del softmax.
 * Con p = softmax(s), la jacobiana es dp_i/ds_j = p_i(delta_ij - p_j), luego
 *   dL/ds_j = p_j * ( dL/dp_j - sum_k p_k * dL/dp_k ).
 * El termino sum_k p_k*dP_k es un producto escalar por fila, que se calcula
 * con una reduccion en memoria compartida. */
__global__ void softmax_backward_kernel(const float *__restrict__ P,
                                        const float *__restrict__ dP,
                                        float *__restrict__ dS,
                                        int n_cols, float scale)
{
    __shared__ float red[SOFTMAX_THREADS];
    const size_t base = (size_t)blockIdx.x * n_cols;
    const int tid = threadIdx.x;

    float dot = 0.0f;
    for (int j = tid; j < n_cols; j += blockDim.x) dot += P[base + j] * dP[base + j];
    red[tid] = dot;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float row_dot = red[0];

    for (int j = tid; j < n_cols; j += blockDim.x)
        dS[base + j] = scale * P[base + j] * (dP[base + j] - row_dot);
}

/* ==========================================================================
 *  Reordenamientos de memoria
 * ========================================================================*/

/* qkv[b, t, j*D + h*Dh + d]  ->  q/k/v[b, h, t, d]
 * Se separa en tensores distintos y se permutan (t,h) para que los Dh valores
 * de una cabeza en un token queden contiguos, que es lo que hace que las
 * lecturas de los kernels bmm_* sean coalescidas. */
__global__ void split_qkv_kernel(const float *__restrict__ qkv,
                                 float *__restrict__ q,
                                 float *__restrict__ k,
                                 float *__restrict__ v,
                                 int B, int T, int H, int Dh)
{
    const int D = H * Dh;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * T * D;
    if (idx >= total) return;

    const int d = idx % Dh;
    const int h = (idx / Dh) % H;
    const int t = (idx / D) % T;
    const int b = idx / (T * D);

    const size_t src = ((size_t)b * T + t) * (3 * D) + (size_t)h * Dh + d;
    const size_t dst = (((size_t)b * H + h) * T + t) * Dh + d;
    q[dst] = qkv[src];
    k[dst] = qkv[src + D];
    v[dst] = qkv[src + 2 * D];
}

/* Inversa exacta de split_qkv, para propagar dq/dk/dv de vuelta a dqkv. */
__global__ void merge_qkv_grad_kernel(const float *__restrict__ dq,
                                      const float *__restrict__ dk,
                                      const float *__restrict__ dv,
                                      float *__restrict__ dqkv,
                                      int B, int T, int H, int Dh)
{
    const int D = H * Dh;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * T * D;
    if (idx >= total) return;

    const int d = idx % Dh;
    const int h = (idx / Dh) % H;
    const int t = (idx / D) % T;
    const int b = idx / (T * D);

    const size_t dst = ((size_t)b * T + t) * (3 * D) + (size_t)h * Dh + d;
    const size_t src = (((size_t)b * H + h) * T + t) * Dh + d;
    dqkv[dst]             = dq[src];
    dqkv[dst + D]         = dk[src];
    dqkv[dst + 2 * D]     = dv[src];
}

/* attn[b,h,t,d] -> y[b,t,h*Dh+d]: reune las cabezas antes de la proyeccion. */
__global__ void merge_heads_kernel(const float *__restrict__ attn,
                                   float *__restrict__ y,
                                   int B, int T, int H, int Dh)
{
    const int D = H * Dh;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T * D) return;

    const int d = idx % Dh;
    const int h = (idx / Dh) % H;
    const int t = (idx / D) % T;
    const int b = idx / (T * D);

    y[((size_t)b * T + t) * D + h * Dh + d] =
        attn[(((size_t)b * H + h) * T + t) * Dh + d];
}

/* Inversa de merge_heads. */
__global__ void split_heads_kernel(const float *__restrict__ dy,
                                   float *__restrict__ dattn,
                                   int B, int T, int H, int Dh)
{
    const int D = H * Dh;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T * D) return;

    const int d = idx % Dh;
    const int h = (idx / Dh) % H;
    const int t = (idx / D) % T;
    const int b = idx / (T * D);

    dattn[(((size_t)b * H + h) * T + t) * Dh + d] =
        dy[((size_t)b * T + t) * D + h * Dh + d];
}

/* ------------------------------- lanzadores ------------------------------ */

void bmm_nt(const float *A, const float *B, float *C,
            int batch, int M, int N, int K,
            int sA, int sB, int sC, float alpha, AttnMem mem, cudaStream_t stream)
{
    dim3 block(TILE, TILE);
    dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE), batch);
    if (mem == ATTN_SHARED)
        bmm_nt_shared_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, sA, sB, sC, alpha);
    else
        bmm_nt_global_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, sA, sB, sC, alpha);
    CUDA_CHECK_KERNEL();
}

void bmm_nn(const float *A, const float *B, float *C,
            int batch, int M, int N, int K,
            int sA, int sB, int sC, float alpha, AttnMem mem, cudaStream_t stream)
{
    dim3 block(TILE, TILE);
    dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE), batch);
    if (mem == ATTN_SHARED)
        bmm_nn_shared_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, sA, sB, sC, alpha);
    else
        bmm_nn_global_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, sA, sB, sC, alpha);
    CUDA_CHECK_KERNEL();
}

void bmm_tn(const float *A, const float *B, float *C,
            int batch, int M, int N, int K,
            int sA, int sB, int sC, float alpha, AttnMem mem, cudaStream_t stream)
{
    dim3 block(TILE, TILE);
    dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE), batch);
    if (mem == ATTN_SHARED)
        bmm_tn_shared_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, sA, sB, sC, alpha);
    else
        bmm_tn_global_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, sA, sB, sC, alpha);
    CUDA_CHECK_KERNEL();
}

void softmax_rows(const float *x, float *y, int n_rows, int n_cols, cudaStream_t stream)
{
    softmax_rows_kernel<<<n_rows, SOFTMAX_THREADS, 0, stream>>>(x, y, n_cols);
    CUDA_CHECK_KERNEL();
}

void softmax_backward(const float *P, const float *dP, float *dS,
                      int n_rows, int n_cols, float scale, cudaStream_t stream)
{
    softmax_backward_kernel<<<n_rows, SOFTMAX_THREADS, 0, stream>>>(P, dP, dS, n_cols, scale);
    CUDA_CHECK_KERNEL();
}

void split_qkv(const float *qkv, float *q, float *k, float *v,
               int B, int T, int H, int Dh, cudaStream_t stream)
{
    const int total = B * T * H * Dh, threads = 256;
    split_qkv_kernel<<<ceil_div(total, threads), threads, 0, stream>>>(qkv, q, k, v, B, T, H, Dh);
    CUDA_CHECK_KERNEL();
}

void merge_qkv_grad(const float *dq, const float *dk, const float *dv, float *dqkv,
                    int B, int T, int H, int Dh, cudaStream_t stream)
{
    const int total = B * T * H * Dh, threads = 256;
    merge_qkv_grad_kernel<<<ceil_div(total, threads), threads, 0, stream>>>(dq, dk, dv, dqkv, B, T, H, Dh);
    CUDA_CHECK_KERNEL();
}

void merge_heads(const float *attn, float *y, int B, int T, int H, int Dh, cudaStream_t stream)
{
    const int total = B * T * H * Dh, threads = 256;
    merge_heads_kernel<<<ceil_div(total, threads), threads, 0, stream>>>(attn, y, B, T, H, Dh);
    CUDA_CHECK_KERNEL();
}

void split_heads(const float *dy, float *dattn, int B, int T, int H, int Dh, cudaStream_t stream)
{
    const int total = B * T * H * Dh, threads = 256;
    split_heads_kernel<<<ceil_div(total, threads), threads, 0, stream>>>(dy, dattn, B, T, H, Dh);
    CUDA_CHECK_KERNEL();
}
