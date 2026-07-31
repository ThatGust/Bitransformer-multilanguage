/* ============================================================================
 * layernorm.cu
 *
 * Eleccion de grid/block: un bloque por fila (n_rows = B*T bloques) y 128
 * hilos por bloque. D vale 64 en este modelo, asi que un solo bloque cubre la
 * fila entera y las dos reducciones (media y varianza) se resuelven en memoria
 * compartida sin necesidad de atomics ni de un segundo lanzamiento. Los hilos
 * recorren la fila con paso blockDim.x, de modo que hilos consecutivos leen
 * direcciones consecutivas (accesos coalescidos).
 * ==========================================================================*/
#include "common.cuh"
#include "layernorm.cuh"

#define LN_THREADS 128

__global__ void layernorm_forward_kernel(const float *__restrict__ x,
                                         const float *__restrict__ gamma,
                                         const float *__restrict__ beta,
                                         float *__restrict__ y,
                                         float *__restrict__ mean,
                                         float *__restrict__ rstd,
                                         int D, float eps)
{
    __shared__ float red[LN_THREADS];
    const size_t base = (size_t)blockIdx.x * D;
    const int tid = threadIdx.x;

    /* --- media --- */
    float s = 0.0f;
    for (int j = tid; j < D; j += blockDim.x) s += x[base + j];
    red[tid] = s;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (tid < k) red[tid] += red[tid + k];
        __syncthreads();
    }
    const float mu = red[0] / D;
    __syncthreads();

    /* --- varianza --- */
    float v = 0.0f;
    for (int j = tid; j < D; j += blockDim.x) { float d = x[base + j] - mu; v += d * d; }
    red[tid] = v;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (tid < k) red[tid] += red[tid + k];
        __syncthreads();
    }
    const float rs = rsqrtf(red[0] / D + eps);

    if (tid == 0) { mean[blockIdx.x] = mu; rstd[blockIdx.x] = rs; }

    for (int j = tid; j < D; j += blockDim.x)
        y[base + j] = gamma[j] * ((x[base + j] - mu) * rs) + beta[j];
}

/* ---------------------------------------------------------------------------
 * Backward respecto de la ENTRADA.
 *
 * Derivacion (por fila, con g = gamma, r = rstd, xhat = (x-mu)*r):
 *   dL/dxhat_j = dy_j * g_j
 *   dL/dvar    = sum_j dL/dxhat_j * (x_j-mu) * (-1/2)(var+eps)^{-3/2}
 *   dL/dmu     = sum_j dL/dxhat_j * (-r)      [el termino via var se cancela
 *                                              porque sum_j (x_j-mu) = 0]
 * Sustituyendo y agrupando se llega a la forma compacta que se implementa:
 *   dx_j = r * ( dxhat_j - mean_k(dxhat_k) - xhat_j * mean_k(dxhat_k*xhat_k) )
 *
 * Es decir, dos productos escalares por fila: la suma de dxhat y la suma de
 * dxhat*xhat. Ambos se calculan con una unica pasada y dos reducciones.
 * -------------------------------------------------------------------------*/
__global__ void layernorm_backward_input_kernel(const float *__restrict__ dy,
                                                const float *__restrict__ x,
                                                const float *__restrict__ gamma,
                                                const float *__restrict__ mean,
                                                const float *__restrict__ rstd,
                                                float *__restrict__ dx,
                                                int D)
{
    __shared__ float red1[LN_THREADS];
    __shared__ float red2[LN_THREADS];
    const size_t base = (size_t)blockIdx.x * D;
    const int tid = threadIdx.x;
    const float mu = mean[blockIdx.x];
    const float r  = rstd[blockIdx.x];

    float s1 = 0.0f, s2 = 0.0f;
    for (int j = tid; j < D; j += blockDim.x) {
        float xhat  = (x[base + j] - mu) * r;
        float dxhat = dy[base + j] * gamma[j];
        s1 += dxhat;
        s2 += dxhat * xhat;
    }
    red1[tid] = s1; red2[tid] = s2;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (tid < k) { red1[tid] += red1[tid + k]; red2[tid] += red2[tid + k]; }
        __syncthreads();
    }
    const float m1 = red1[0] / D;   /* mean_k(dxhat_k)        */
    const float m2 = red2[0] / D;   /* mean_k(dxhat_k*xhat_k) */

    for (int j = tid; j < D; j += blockDim.x) {
        float xhat  = (x[base + j] - mu) * r;
        float dxhat = dy[base + j] * gamma[j];
        dx[base + j] = r * (dxhat - m1 - xhat * m2);
    }
}

/* ---------------------------------------------------------------------------
 * Backward respecto de los PARAMETROS:
 *   dgamma_j = sum_i dy[i,j] * xhat[i,j]
 *   dbeta_j  = sum_i dy[i,j]
 * Reduccion sobre las filas: un bloque por columna j. Se prefiere esto a usar
 * atomicAdd desde el kernel anterior porque evita la contencion (n_rows puede
 * ser de miles) y hace el resultado deterministico entre corridas.
 * -------------------------------------------------------------------------*/
__global__ void layernorm_backward_params_kernel(const float *__restrict__ dy,
                                                 const float *__restrict__ x,
                                                 const float *__restrict__ mean,
                                                 const float *__restrict__ rstd,
                                                 float *__restrict__ dgamma,
                                                 float *__restrict__ dbeta,
                                                 int n_rows, int D)
{
    __shared__ float rg[LN_THREADS];
    __shared__ float rb[LN_THREADS];
    const int j = blockIdx.x;
    const int tid = threadIdx.x;

    float sg = 0.0f, sb = 0.0f;
    for (int i = tid; i < n_rows; i += blockDim.x) {
        float d = dy[(size_t)i * D + j];
        float xhat = (x[(size_t)i * D + j] - mean[i]) * rstd[i];
        sg += d * xhat;
        sb += d;
    }
    rg[tid] = sg; rb[tid] = sb;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (tid < k) { rg[tid] += rg[tid + k]; rb[tid] += rb[tid + k]; }
        __syncthreads();
    }
    if (tid == 0) { dgamma[j] = rg[0]; dbeta[j] = rb[0]; }
}

/* ------------------------------- lanzadores ------------------------------ */

void layernorm_forward(const float *x, const float *gamma, const float *beta,
                       float *y, float *mean, float *rstd,
                       int n_rows, int D, float eps, cudaStream_t stream)
{
    layernorm_forward_kernel<<<n_rows, LN_THREADS, 0, stream>>>(x, gamma, beta, y, mean, rstd, D, eps);
    CUDA_CHECK_KERNEL();
}

void layernorm_backward(const float *dy, const float *x, const float *gamma,
                        const float *mean, const float *rstd,
                        float *dx, float *dgamma, float *dbeta,
                        int n_rows, int D, cudaStream_t stream)
{
    layernorm_backward_input_kernel<<<n_rows, LN_THREADS, 0, stream>>>(dy, x, gamma, mean, rstd, dx, D);
    CUDA_CHECK_KERNEL();
    layernorm_backward_params_kernel<<<D, LN_THREADS, 0, stream>>>(dy, x, mean, rstd, dgamma, dbeta, n_rows, D);
    CUDA_CHECK_KERNEL();
}
