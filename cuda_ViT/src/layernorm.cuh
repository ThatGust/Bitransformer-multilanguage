/* ============================================================================
 * layernorm.cuh -- Normalizacion por capa sobre la ultima dimension.
 *
 * Forward, para cada fila x de longitud D:
 *   mu    = (1/D) sum_j x_j
 *   var   = (1/D) sum_j (x_j - mu)^2
 *   xhat  = (x - mu) / sqrt(var + eps)
 *   y     = gamma * xhat + beta
 *
 * Se guardan mean y rstd = 1/sqrt(var+eps) para no recalcularlos en el
 * backward (es el compromiso clasico memoria-por-computo).
 * ==========================================================================*/
#ifndef LAYERNORM_CUH
#define LAYERNORM_CUH

void layernorm_forward(const float *x, const float *gamma, const float *beta,
                       float *y, float *mean, float *rstd,
                       int n_rows, int D, float eps, cudaStream_t stream = 0);

/* Backward analitico. Ver la derivacion completa en layernorm.cu.
 * dgamma y dbeta se ESCRIBEN (no se acumulan): el llamador usa buffers de
 * gradiente propios de cada parametro que se ponen a cero cada paso. */
void layernorm_backward(const float *dy, const float *x, const float *gamma,
                        const float *mean, const float *rstd,
                        float *dx, float *dgamma, float *dbeta,
                        int n_rows, int D, cudaStream_t stream = 0);

#endif /* LAYERNORM_CUH */
