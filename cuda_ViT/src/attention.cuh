/* ============================================================================
 * attention.cuh -- Multi-Head Self-Attention.
 *
 * Este es el modulo donde vive el EXPERIMENTO 2 del trabajo: las dos
 * multiplicaciones matriciales por cabeza (QK^T y P*V) estan implementadas
 * dos veces, una usando solo memoria global y otra con tiling en memoria
 * compartida, seleccionables en tiempo de ejecucion con AttnMem.
 *
 * Layout de tensores (todo row-major, contiguo):
 *   qkv     [B, T, 3*D]   salida de la proyeccion lineal; la columna c se
 *                         interpreta como c = j*D + h*Dh + d con j in {q,k,v}
 *   q,k,v   [B, H, T, Dh] tras split_qkv; esta permutacion pone los Dh de una
 *                         cabeza contiguos, que es lo que necesitan los
 *                         productos por cabeza para leer coalescido
 *   scores  [B, H, T, T]
 *   probs   [B, H, T, T]
 *   attn    [B, H, T, Dh]
 * ==========================================================================*/
#ifndef ATTENTION_CUH
#define ATTENTION_CUH

/* Selector de la variante de memoria del kernel critico. */
enum AttnMem { ATTN_GLOBAL = 0, ATTN_SHARED = 1 };

/* ---- Multiplicaciones batcheadas (una matriz independiente por (b,h)) ----
 * batch = B*H. sA/sB/sC son los strides en elementos entre matrices sucesivas.
 * alpha escala el resultado (se usa para el 1/sqrt(Dh) de la atencion).       */

/* C[M,N] = alpha * A[M,K] * B[N,K]^T   -- usado por QK^T */
void bmm_nt(const float *A, const float *B, float *C,
            int batch, int M, int N, int K,
            int sA, int sB, int sC, float alpha, AttnMem mem,
            cudaStream_t stream = 0);

/* C[M,N] = alpha * A[M,K] * B[K,N]     -- usado por P*V */
void bmm_nn(const float *A, const float *B, float *C,
            int batch, int M, int N, int K,
            int sA, int sB, int sC, float alpha, AttnMem mem,
            cudaStream_t stream = 0);

/* C[M,N] = alpha * A[K,M]^T * B[K,N]   -- usado por dV y dK */
void bmm_tn(const float *A, const float *B, float *C,
            int batch, int M, int N, int K,
            int sA, int sB, int sC, float alpha, AttnMem mem,
            cudaStream_t stream = 0);

/* ---- Softmax por filas de longitud n_cols (n_rows = B*H*T filas) ---- */
void softmax_rows(const float *x, float *y, int n_rows, int n_cols,
                  cudaStream_t stream = 0);

/* Backward del softmax fila a fila:
 *   dS[i,j] = scale * P[i,j] * ( dP[i,j] - sum_k P[i,k]*dP[i,k] )
 * scale absorbe el factor 1/sqrt(Dh) aplicado en el forward, de modo que dS
 * ya es el gradiente respecto de Q*K^T (sin escalar). */
void softmax_backward(const float *P, const float *dP, float *dS,
                      int n_rows, int n_cols, float scale,
                      cudaStream_t stream = 0);

/* ---- Reordenamientos entre [B,T,3D] / [B,H,T,Dh] / [B,T,D] ---- */
void split_qkv(const float *qkv, float *q, float *k, float *v,
               int B, int T, int H, int Dh, cudaStream_t stream = 0);
void merge_qkv_grad(const float *dq, const float *dk, const float *dv, float *dqkv,
                    int B, int T, int H, int Dh, cudaStream_t stream = 0);
void merge_heads(const float *attn, float *y, int B, int T, int H, int Dh,
                 cudaStream_t stream = 0);
void split_heads(const float *dy, float *dattn, int B, int T, int H, int Dh,
                 cudaStream_t stream = 0);

#endif /* ATTENTION_CUH */
