/* ============================================================================
 * patch_embed.cu
 *
 * Nota de diseno: la proyeccion lineal del parche NO tiene kernel propio.
 * im2patch reorganiza la imagen en una matriz [B*n_patches, patch_dim] y la
 * proyeccion es entonces un gemm_nn ordinario contra W_patch[patch_dim, D].
 * Reutilizar el GEMM tileado en vez de escribir un kernel especializado evita
 * duplicar codigo y hace que el patch embedding se beneficie del mismo tiling
 * que el resto de las capas lineales.
 *
 * Embeddings posicionales APRENDIBLES (no seno/coseno fijos). Razon: con solo
 * 5, 17 o 50 posiciones y un vocabulario de posiciones fijo y pequeno, la
 * tabla aprendible cuesta apenas T*D parametros (3200 en el peor caso, ~4% del
 * modelo) y evita tener que justificar la eleccion de frecuencias del esquema
 * sinusoidal, que esta pensado para secuencias de longitud variable. Aqui la
 * longitud es siempre la misma dentro de una configuracion.
 * ==========================================================================*/
#include "common.cuh"
#include "patch_embed.cuh"

#define THREADS 256

/* Un hilo por elemento de la matriz de parches de salida. Se indexa desde la
 * SALIDA hacia la entrada (gather) y no al reves (scatter) para que las
 * escrituras sean contiguas y coalescidas; las lecturas de la imagen quedan
 * con saltos, pero pasan por la cache de solo lectura. */
__global__ void im2patch_kernel(const float *__restrict__ images,
                                float *__restrict__ patches,
                                int B, int img_size, int patch,
                                int grid_side, int patch_dim)
{
    const int n_patches = grid_side * grid_side;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * n_patches * patch_dim) return;

    const int k = idx % patch_dim;              /* posicion dentro del parche */
    const int p = (idx / patch_dim) % n_patches;/* indice del parche          */
    const int b = idx / (patch_dim * n_patches);

    const int pr = p / grid_side, pc = p % grid_side;   /* parche (fila, col) */
    const int kr = k / patch,     kc = k % patch;       /* pixel dentro       */

    const int r = pr * patch + kr;
    const int c = pc * patch + kc;

    patches[idx] = images[((size_t)b * img_size + r) * img_size + c];
}

__global__ void assemble_tokens_kernel(const float *__restrict__ emb,
                                       const float *__restrict__ cls,
                                       const float *__restrict__ pos,
                                       float *__restrict__ tokens,
                                       int B, int n_patches, int D)
{
    const int T = n_patches + 1;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T * D) return;

    const int d = idx % D;
    const int t = (idx / D) % T;
    const int b = idx / (T * D);

    float base = (t == 0) ? cls[d]
                          : emb[((size_t)b * n_patches + (t - 1)) * D + d];
    tokens[idx] = base + pos[(size_t)t * D + d];
}

/* demb: copia directa (la suma con pos no altera el gradiente). */
__global__ void demb_kernel(const float *__restrict__ dtokens,
                            float *__restrict__ demb,
                            int B, int n_patches, int D)
{
    const int T = n_patches + 1;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * n_patches * D) return;

    const int d = idx % D;
    const int p = (idx / D) % n_patches;
    const int b = idx / (n_patches * D);

    demb[idx] = dtokens[(((size_t)b * T) + (p + 1)) * D + d];
}

/* dpos[t,d] = sum_b dtokens[b,t,d] y dcls[d] = sum_b dtokens[b,0,d].
 * Un hilo por (t,d) recorriendo el batch: B es 64, un bucle corto y sin
 * necesidad de atomics. */
__global__ void dpos_dcls_kernel(const float *__restrict__ dtokens,
                                 float *__restrict__ dcls,
                                 float *__restrict__ dpos,
                                 int B, int T, int D)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * D) return;
    const int d = idx % D;
    const int t = idx / D;

    float s = 0.0f;
    for (int b = 0; b < B; ++b) s += dtokens[(((size_t)b * T) + t) * D + d];
    dpos[idx] = s;
    if (t == 0) dcls[d] = s;
}

/* ------------------------------- lanzadores ------------------------------ */

void im2patch(const float *images, float *patches, int B, int img_size, int patch,
              cudaStream_t stream)
{
    const int grid_side = img_size / patch;
    const int patch_dim = patch * patch;
    const int n = B * grid_side * grid_side * patch_dim;
    im2patch_kernel<<<ceil_div(n, THREADS), THREADS, 0, stream>>>(
        images, patches, B, img_size, patch, grid_side, patch_dim);
    CUDA_CHECK_KERNEL();
}

void assemble_tokens(const float *emb, const float *cls, const float *pos,
                     float *tokens, int B, int n_patches, int D, cudaStream_t stream)
{
    const int n = B * (n_patches + 1) * D;
    assemble_tokens_kernel<<<ceil_div(n, THREADS), THREADS, 0, stream>>>(
        emb, cls, pos, tokens, B, n_patches, D);
    CUDA_CHECK_KERNEL();
}

void assemble_tokens_backward(const float *dtokens, float *demb,
                              float *dcls, float *dpos,
                              int B, int n_patches, int D, cudaStream_t stream)
{
    const int T = n_patches + 1;
    int n = B * n_patches * D;
    demb_kernel<<<ceil_div(n, THREADS), THREADS, 0, stream>>>(dtokens, demb, B, n_patches, D);
    CUDA_CHECK_KERNEL();

    n = T * D;
    dpos_dcls_kernel<<<ceil_div(n, THREADS), THREADS, 0, stream>>>(dtokens, dcls, dpos, B, T, D);
    CUDA_CHECK_KERNEL();
}
