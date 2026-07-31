/* ============================================================================
 * gemm.cu -- Kernels de multiplicacion de matrices con tiling en memoria
 *            compartida.
 *
 * Estrategia comun a los tres kernels: cada bloque de 16x16 hilos calcula un
 * tile de 16x16 de la matriz de salida C. El bucle sobre la dimension K se
 * recorre por tiles: en cada iteracion los 256 hilos cooperan para traer un
 * tile de A y uno de B a memoria compartida, se sincronizan, y luego cada hilo
 * acumula 16 productos leyendo solo de memoria compartida.
 *
 * Por que esto es mas rapido que la version directa: sin tiling, cada elemento
 * de A se relee N veces y cada elemento de B se relee M veces desde memoria
 * global. Con tiles de 16x16, cada elemento traido a memoria compartida se
 * reutiliza 16 veces, reduciendo el trafico contra memoria global en un factor
 * de ~16. La memoria compartida tiene una latencia de ~20-30 ciclos frente a
 * los ~400-600 de la memoria global sin cachear.
 *
 * Grid/block: blockDim = (TILE, TILE) = (16,16) = 256 hilos, multiplo del warp.
 *             gridDim  = (ceil(N/16), ceil(M/16)); x recorre columnas para que
 *             hilos consecutivos (mismo threadIdx.y, threadIdx.x creciente)
 *             accedan a direcciones consecutivas de C -> escrituras coalescidas.
 * ==========================================================================*/
#include "common.cuh"
#include "gemm.cuh"

/* --------------------------------------------------------------------------
 * C[M,N] = A[M,K] * B[K,N] + bias[N]
 * ------------------------------------------------------------------------*/
__global__ void gemm_nn_kernel(const float *__restrict__ A,
                               const float *__restrict__ B,
                               const float *__restrict__ bias,
                               float *__restrict__ C,
                               int M, int N, int K)
{
    __shared__ float As[TILE][TILE];
    /* +1 en la segunda dimension de Bs para evitar conflictos de banco: con
     * 32 bancos y stride 16 varios hilos del mismo warp caerian en el mismo
     * banco al leer una columna; el padding rompe esa alineacion. */
    __shared__ float Bs[TILE][TILE + 1];

    const int row = blockIdx.y * TILE + threadIdx.y;   /* fila de C */
    const int col = blockIdx.x * TILE + threadIdx.x;   /* columna de C */
    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        /* Carga cooperativa. Los hilos fuera de rango escriben 0 en vez de
         * saltarse la carga: asi el bucle de acumulacion no necesita guardas
         * y todos los hilos llegan a __syncthreads(). */
        int a_col = t + threadIdx.x;
        int b_row = t + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int i = 0; i < TILE; ++i)
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        /* Segunda barrera: impide que un hilo rapido sobrescriba el tile
         * mientras otro todavia lo esta leyendo en la iteracion actual. */
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc + (bias ? bias[col] : 0.0f);
}

/* --------------------------------------------------------------------------
 * C[M,N] = A[M,K] * B[N,K]^T     (B almacenada por filas de longitud K)
 * Aparece en dX = dY * W^T.
 * ------------------------------------------------------------------------*/
__global__ void gemm_nt_kernel(const float *__restrict__ A,
                               const float *__restrict__ B,
                               float *__restrict__ C,
                               int M, int N, int K)
{
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        int a_col = t + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        /* Bs[n_local][i] = B[bloque_n + n_local, t + i]. La fila de B la indexa
         * threadIdx.y y la dimension contraida threadIdx.x, de modo que hilos
         * consecutivos leen posiciones consecutivas de B (coalescido). */
        int b_row = blockIdx.x * TILE + threadIdx.y;
        int b_col = t + threadIdx.x;
        Bs[threadIdx.y][threadIdx.x] = (b_row < N && b_col < K) ? B[b_row * K + b_col] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int i = 0; i < TILE; ++i)
            acc += As[threadIdx.y][i] * Bs[threadIdx.x][i];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

/* --------------------------------------------------------------------------
 * C[M,N] = A[K,M]^T * B[K,N]
 * Aparece en dW = X^T * dY, donde la dimension contraida K = batch*tokens
 * suele ser grande (miles), por lo que el tiling importa especialmente aqui.
 * ------------------------------------------------------------------------*/
__global__ void gemm_tn_kernel(const float *__restrict__ A,
                               const float *__restrict__ B,
                               float *__restrict__ C,
                               int M, int N, int K)
{
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    const int row = blockIdx.y * TILE + threadIdx.y;   /* indice sobre M */
    const int col = blockIdx.x * TILE + threadIdx.x;   /* indice sobre N */
    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        /* As[i][m] = A[(t+i), row]. Se carga con threadIdx.y recorriendo i
         * para que la lectura de A sea coalescida a lo largo de M. */
        int a_row = t + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (a_row < K && (blockIdx.y * TILE + threadIdx.x) < M)
                                     ? A[a_row * M + blockIdx.y * TILE + threadIdx.x] : 0.0f;
        int b_row = t + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int i = 0; i < TILE; ++i)
            acc += As[i][threadIdx.y] * Bs[i][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

/* --------------------------------------------------------------------------
 * db[n] = sum_{m} dY[m,n].
 * Un bloque por columna de salida; los hilos del bloque recorren las M filas
 * con paso blockDim.x (accesos coalescidos entre hilos consecutivos) y luego
 * se reducen en memoria compartida.
 * ------------------------------------------------------------------------*/
__global__ void bias_backward_kernel(const float *__restrict__ dY,
                                     float *__restrict__ db,
                                     int M, int N)
{
    extern __shared__ float sdata[];
    const int n = blockIdx.x;
    float sum = 0.0f;
    for (int m = threadIdx.x; m < M; m += blockDim.x)
        sum += dY[m * N + n];

    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) db[n] = sdata[0];
}

/* ------------------------------- lanzadores ------------------------------ */

void gemm_nn(const float *A, const float *B, const float *bias, float *C,
             int M, int N, int K, cudaStream_t stream)
{
    dim3 block(TILE, TILE);
    dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
    gemm_nn_kernel<<<grid, block, 0, stream>>>(A, B, bias, C, M, N, K);
    CUDA_CHECK_KERNEL();
}

void gemm_nt(const float *A, const float *B, float *C,
             int M, int N, int K, cudaStream_t stream)
{
    dim3 block(TILE, TILE);
    dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
    gemm_nt_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK_KERNEL();
}

void gemm_tn(const float *A, const float *B, float *C,
             int M, int N, int K, cudaStream_t stream)
{
    dim3 block(TILE, TILE);
    dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
    gemm_tn_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK_KERNEL();
}

void bias_backward(const float *dY, float *db, int M, int N, cudaStream_t stream)
{
    const int threads = 256;
    bias_backward_kernel<<<N, threads, threads * sizeof(float), stream>>>(dY, db, M, N);
    CUDA_CHECK_KERNEL();
}
