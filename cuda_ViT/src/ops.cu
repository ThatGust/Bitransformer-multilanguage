/* ============================================================================
 * ops.cu -- Kernels elementales.
 *
 * Todos los kernels de esta unidad son "element-wise" o reducciones por fila
 * pequenas, es decir, limitados por ancho de banda de memoria y no por
 * computo. Por eso la configuracion es siempre la misma y sencilla: bloques
 * unidimensionales de 256 hilos (8 warps, buen equilibrio entre ocupacion y
 * registros) y una grilla de ceil(n/256) bloques, con un hilo por elemento y
 * accesos perfectamente coalescidos.
 * ==========================================================================*/
#include "common.cuh"
#include "ops.cuh"

#include <cmath>

#define THREADS 256

/* Constantes de la aproximacion tanh de GELU. */
#define GELU_C  0.7978845608028654f   /* sqrt(2/pi) */
#define GELU_A  0.044715f

__global__ void gelu_forward_kernel(const float *__restrict__ x,
                                    float *__restrict__ y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float u = GELU_C * (v + GELU_A * v * v * v);
    y[i] = 0.5f * v * (1.0f + tanhf(u));
}

/* Derivada de GELU. Con u = c(x + a x^3) y t = tanh(u):
 *   dy/dx = 0.5(1+t) + 0.5*x*(1-t^2)*du/dx,   du/dx = c(1 + 3a x^2)
 * El termino (1-t^2) es sech^2(u), la derivada de la tangente hiperbolica. */
__global__ void gelu_backward_kernel(const float *__restrict__ x,
                                     const float *__restrict__ dy,
                                     float *__restrict__ dx, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float u = GELU_C * (v + GELU_A * v * v * v);
    float t = tanhf(u);
    float dudx = GELU_C * (1.0f + 3.0f * GELU_A * v * v);
    dx[i] = dy[i] * (0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * dudx);
}

__global__ void add_forward_kernel(const float *__restrict__ a,
                                   const float *__restrict__ b,
                                   float *__restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void add_inplace_kernel(float *__restrict__ a,
                                   const float *__restrict__ b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

/* Softmax + entropia cruzada. Un bloque por muestra; C = 10 es tan pequeno que
 * un unico hilo por bloque hace el trabajo sin perder nada: el coste real esta
 * dominado por la latencia de acceso, no por la aritmetica. Se resta el maximo
 * antes de exponenciar por estabilidad numerica. */
__global__ void softmax_ce_forward_kernel(const float *__restrict__ logits,
                                          const int *__restrict__ labels,
                                          float *__restrict__ probs,
                                          float *__restrict__ loss,
                                          int C)
{
    const int b = blockIdx.x;
    const float *lg = logits + (size_t)b * C;
    float *pr = probs + (size_t)b * C;

    float mx = -INFINITY;
    for (int j = 0; j < C; ++j) mx = fmaxf(mx, lg[j]);
    float sum = 0.0f;
    for (int j = 0; j < C; ++j) { float e = expf(lg[j] - mx); pr[j] = e; sum += e; }
    float inv = 1.0f / sum;
    for (int j = 0; j < C; ++j) pr[j] *= inv;

    /* -log p_y calculado como (max + log sum exp) - logit_y: evita evaluar
     * log() sobre una probabilidad que puede haber caido a 0 en float. */
    loss[b] = (mx + logf(sum)) - lg[labels[b]];
}

__global__ void softmax_ce_backward_kernel(const float *__restrict__ probs,
                                           const int *__restrict__ labels,
                                           float *__restrict__ dlogits,
                                           int B, int C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * C) return;
    int b = i / C, j = i % C;
    float onehot = (j == labels[b]) ? 1.0f : 0.0f;
    dlogits[i] = (probs[i] - onehot) / (float)B;
}

__global__ void count_correct_kernel(const float *__restrict__ probs,
                                     const int *__restrict__ labels,
                                     int *__restrict__ d_correct, int B, int C)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const float *pr = probs + (size_t)b * C;
    int best = 0;
    float bv = pr[0];
    for (int j = 1; j < C; ++j) if (pr[j] > bv) { bv = pr[j]; best = j; }
    if (best == labels[b]) atomicAdd(d_correct, 1);
}

__global__ void fill_zero_kernel(float *__restrict__ x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.0f;
}

/* Adam. bc1 y bc2 (las correcciones de sesgo) se pasan ya calculadas desde el
 * host para no repetir powf() en cada uno de los ~72k hilos. */
__global__ void adam_kernel(float *__restrict__ param,
                            const float *__restrict__ grad,
                            float *__restrict__ m, float *__restrict__ v,
                            int n, float lr, float beta1, float beta2,
                            float eps, float weight_decay, float bc1, float bc2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grad[i];
    float mi = beta1 * m[i] + (1.0f - beta1) * g;
    float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;
    float mhat = mi / bc1;
    float vhat = vi / bc2;
    /* weight decay desacoplado (AdamW): se aplica sobre el parametro y no
     * sobre el gradiente, para que no entre en las estimaciones de momento. */
    param[i] -= lr * (mhat / (sqrtf(vhat) + eps) + weight_decay * param[i]);
}

/* ------------------------------- lanzadores ------------------------------ */

void gelu_forward(const float *x, float *y, int n, cudaStream_t stream)
{
    gelu_forward_kernel<<<ceil_div(n, THREADS), THREADS, 0, stream>>>(x, y, n);
    CUDA_CHECK_KERNEL();
}

void gelu_backward(const float *x, const float *dy, float *dx, int n, cudaStream_t stream)
{
    gelu_backward_kernel<<<ceil_div(n, THREADS), THREADS, 0, stream>>>(x, dy, dx, n);
    CUDA_CHECK_KERNEL();
}

void add_forward(const float *a, const float *b, float *out, int n, cudaStream_t stream)
{
    add_forward_kernel<<<ceil_div(n, THREADS), THREADS, 0, stream>>>(a, b, out, n);
    CUDA_CHECK_KERNEL();
}

void add_inplace(float *a, const float *b, int n, cudaStream_t stream)
{
    add_inplace_kernel<<<ceil_div(n, THREADS), THREADS, 0, stream>>>(a, b, n);
    CUDA_CHECK_KERNEL();
}

void fill_zero(float *x, int n, cudaStream_t stream)
{
    fill_zero_kernel<<<ceil_div(n, THREADS), THREADS, 0, stream>>>(x, n);
    CUDA_CHECK_KERNEL();
}

void softmax_cross_entropy_forward(const float *logits, const int *labels,
                                   float *probs, float *loss_per_sample,
                                   int B, int C, cudaStream_t stream)
{
    softmax_ce_forward_kernel<<<B, 1, 0, stream>>>(logits, labels, probs, loss_per_sample, C);
    CUDA_CHECK_KERNEL();
}

void softmax_cross_entropy_backward(const float *probs, const int *labels,
                                    float *dlogits, int B, int C, cudaStream_t stream)
{
    int n = B * C;
    softmax_ce_backward_kernel<<<ceil_div(n, THREADS), THREADS, 0, stream>>>(probs, labels, dlogits, B, C);
    CUDA_CHECK_KERNEL();
}

void count_correct(const float *probs, const int *labels, int *d_correct,
                   int B, int C, cudaStream_t stream)
{
    count_correct_kernel<<<ceil_div(B, THREADS), THREADS, 0, stream>>>(probs, labels, d_correct, B, C);
    CUDA_CHECK_KERNEL();
}

void adam_step(float *param, const float *grad, float *m, float *v,
               int n, float lr, float beta1, float beta2, float eps,
               float weight_decay, int t, cudaStream_t stream)
{
    const float bc1 = 1.0f - std::pow(beta1, (float)t);
    const float bc2 = 1.0f - std::pow(beta2, (float)t);
    adam_kernel<<<ceil_div(n, THREADS), THREADS, 0, stream>>>(
        param, grad, m, v, n, lr, beta1, beta2, eps, weight_decay, bc1, bc2);
    CUDA_CHECK_KERNEL();
}
