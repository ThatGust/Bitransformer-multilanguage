/* ============================================================================
 * vit.cu -- Forward y backward completos del Vision Transformer.
 *
 * El backward es MANUAL: para cada capa se aplica la regla de la cadena con la
 * derivada analitica cerrada, en orden inverso al forward. No hay grafo de
 * autodiff. Cada bloque de codigo del backward lleva escrita la formula que
 * implementa.
 *
 * Regla que se repite en toda capa lineal Y = X*W + b, con X[M,K], W[K,N]:
 *     dX = dY * W^T      -> gemm_nt
 *     dW = X^T * dY      -> gemm_tn
 *     db = columnas de dY sumadas
 * ==========================================================================*/
#include "common.cuh"
#include "vit.cuh"
#include "gemm.cuh"
#include "attention.cuh"
#include "layernorm.cuh"
#include "patch_embed.cuh"
#include "ops.cuh"

#include <random>
#include <vector>
#include <cstring>
#include <cmath>

/* --------------------------------------------------------------------------
 * Kernels locales: extraccion del token [CLS] y su gradiente.
 * ------------------------------------------------------------------------*/

/* cls_feat[b,:] = tokens[b,0,:] */
__global__ void extract_cls_kernel(const float *__restrict__ tokens,
                                   float *__restrict__ cls_feat,
                                   int B, int T, int D)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * D) return;
    int d = i % D, b = i / D;
    cls_feat[i] = tokens[((size_t)b * T) * D + d];
}

/* Inversa: el gradiente solo entra por la posicion 0; el resto de los tokens
 * no participan en la clasificacion, asi que su gradiente por esta rama es 0.
 * El kernel escribe TODO dtokens (ceros incluidos) para no depender de un
 * memset previo. */
__global__ void scatter_cls_grad_kernel(const float *__restrict__ dcls_feat,
                                        float *__restrict__ dtokens,
                                        int B, int T, int D)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * T * D) return;
    int d = i % D, t = (i / D) % T, b = i / (T * D);
    dtokens[i] = (t == 0) ? dcls_feat[(size_t)b * D + d] : 0.0f;
}

/* --------------------------------------------------------------------------
 * Reserva de memoria y particion del bloque de parametros.
 * ------------------------------------------------------------------------*/

namespace {

struct ParamSpec { float **slot_p; float **slot_g; int size; int fan_in; int init; };
/* init: 0 = normal(0, 1/sqrt(fan_in))  (matrices de peso)
 *       1 = cero                       (sesgos, beta de LayerNorm)
 *       2 = uno                        (gamma de LayerNorm)
 *       3 = normal(0, 0.02)            (CLS y embeddings posicionales)      */

inline void *dmalloc(size_t bytes)
{
    void *ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return ptr;
}

inline float *dfloat(size_t n) { return (float *)dmalloc(n * sizeof(float)); }

} /* namespace anonimo */

void vit_init(ViT &m, const ViTConfig &cfg, unsigned long long seed)
{
    m.cfg = cfg;
    const int B  = cfg.batch;
    const int T  = cfg.n_tokens();
    const int D  = cfg.d_model;
    const int H  = cfg.n_heads;
    const int Dh = cfg.head_dim();
    const int Hd = cfg.mlp_hidden;
    const int NP = cfg.n_patches();
    const int PD = cfg.patch_dim();
    const int C  = cfg.n_classes;
    const int L  = cfg.n_blocks;

    /* --- arreglos de punteros por bloque --- */
    auto alloc_pp = [&](float **&a) { a = new float *[L]; };
    alloc_pp(m.p.ln1_g); alloc_pp(m.p.ln1_b); alloc_pp(m.p.Wqkv); alloc_pp(m.p.bqkv);
    alloc_pp(m.p.Wproj); alloc_pp(m.p.bproj); alloc_pp(m.p.ln2_g); alloc_pp(m.p.ln2_b);
    alloc_pp(m.p.Wfc1);  alloc_pp(m.p.bfc1);  alloc_pp(m.p.Wfc2);  alloc_pp(m.p.bfc2);
    alloc_pp(m.g.ln1_g); alloc_pp(m.g.ln1_b); alloc_pp(m.g.Wqkv); alloc_pp(m.g.bqkv);
    alloc_pp(m.g.Wproj); alloc_pp(m.g.bproj); alloc_pp(m.g.ln2_g); alloc_pp(m.g.ln2_b);
    alloc_pp(m.g.Wfc1);  alloc_pp(m.g.bfc1);  alloc_pp(m.g.Wfc2);  alloc_pp(m.g.bfc2);

    /* --- se declara el layout de parametros como una lista ordenada --- */
    std::vector<ParamSpec> spec;
    auto add = [&](float **pp, float **gg, int size, int fan_in, int init) {
        spec.push_back({pp, gg, size, fan_in, init});
    };

    add(&m.p.W_patch, &m.g.W_patch, PD * D, PD, 0);
    add(&m.p.b_patch, &m.g.b_patch, D,      1,  1);
    add(&m.p.cls,     &m.g.cls,     D,      1,  3);
    add(&m.p.pos,     &m.g.pos,     T * D,  1,  3);
    for (int l = 0; l < L; ++l) {
        add(&m.p.ln1_g[l], &m.g.ln1_g[l], D,          1,  2);
        add(&m.p.ln1_b[l], &m.g.ln1_b[l], D,          1,  1);
        add(&m.p.Wqkv[l],  &m.g.Wqkv[l],  D * 3 * D,  D,  0);
        add(&m.p.bqkv[l],  &m.g.bqkv[l],  3 * D,      1,  1);
        add(&m.p.Wproj[l], &m.g.Wproj[l], D * D,      D,  0);
        add(&m.p.bproj[l], &m.g.bproj[l], D,          1,  1);
        add(&m.p.ln2_g[l], &m.g.ln2_g[l], D,          1,  2);
        add(&m.p.ln2_b[l], &m.g.ln2_b[l], D,          1,  1);
        add(&m.p.Wfc1[l],  &m.g.Wfc1[l],  D * Hd,     D,  0);
        add(&m.p.bfc1[l],  &m.g.bfc1[l],  Hd,         1,  1);
        add(&m.p.Wfc2[l],  &m.g.Wfc2[l],  Hd * D,     Hd, 0);
        add(&m.p.bfc2[l],  &m.g.bfc2[l],  D,          1,  1);
    }
    add(&m.p.lnf_g, &m.g.lnf_g, D,     1, 2);
    add(&m.p.lnf_b, &m.g.lnf_b, D,     1, 1);
    add(&m.p.Whead, &m.g.Whead, D * C, D, 0);
    add(&m.p.bhead, &m.g.bhead, C,     1, 1);

    int total = 0;
    for (const auto &s : spec) total += s.size;
    m.n_params = total;

    m.params = dfloat(total);
    m.grads  = dfloat(total);
    m.adam_m = dfloat(total);
    m.adam_v = dfloat(total);
    CUDA_CHECK(cudaMemset(m.adam_m, 0, (size_t)total * sizeof(float)));
    CUDA_CHECK(cudaMemset(m.adam_v, 0, (size_t)total * sizeof(float)));

    /* --- inicializacion en host y una unica copia a device --- */
    std::vector<float> h(total);
    std::mt19937 rng((unsigned int)seed);
    int off = 0;
    for (const auto &s : spec) {
        *(s.slot_p) = m.params + off;
        *(s.slot_g) = m.grads  + off;
        switch (s.init) {
            case 0: {
                /* Inicializacion tipo Xavier/He escalada por 1/sqrt(fan_in):
                 * mantiene la varianza de las activaciones cerca de 1 al
                 * atravesar la capa, evitando que la senal explote o se apague
                 * al apilar bloques. */
                std::normal_distribution<float> nd(0.0f, 1.0f / std::sqrt((float)s.fan_in));
                for (int i = 0; i < s.size; ++i) h[off + i] = nd(rng);
                break;
            }
            case 1: for (int i = 0; i < s.size; ++i) h[off + i] = 0.0f; break;
            case 2: for (int i = 0; i < s.size; ++i) h[off + i] = 1.0f; break;
            case 3: {
                std::normal_distribution<float> nd(0.0f, 0.02f);
                for (int i = 0; i < s.size; ++i) h[off + i] = nd(rng);
                break;
            }
        }
        off += s.size;
    }
    CUDA_CHECK(cudaMemcpy(m.params, h.data(), (size_t)total * sizeof(float), cudaMemcpyHostToDevice));

    /* --- activaciones --- */
    const int BTD  = B * T * D;
    const int BHTT = B * H * T * T;

    m.d_images = dfloat((size_t)B * cfg.img_size * cfg.img_size);
    CUDA_CHECK(cudaMalloc(&m.d_labels, (size_t)B * sizeof(int)));
    m.patches = dfloat((size_t)B * NP * PD);
    m.emb     = dfloat((size_t)B * NP * D);
    m.tokens  = dfloat(BTD);

    m.blocks = new BlockAct[L];
    for (int l = 0; l < L; ++l) {
        BlockAct &a = m.blocks[l];
        a.ln1_out = dfloat(BTD); a.ln1_mean = dfloat((size_t)B * T); a.ln1_rstd = dfloat((size_t)B * T);
        a.qkv = dfloat((size_t)BTD * 3);
        a.q = dfloat(BTD); a.k = dfloat(BTD); a.v = dfloat(BTD);
        a.scores = dfloat(BHTT); a.probs = dfloat(BHTT);
        a.attn = dfloat(BTD); a.merged = dfloat(BTD); a.proj = dfloat(BTD); a.res1 = dfloat(BTD);
        a.ln2_out = dfloat(BTD); a.ln2_mean = dfloat((size_t)B * T); a.ln2_rstd = dfloat((size_t)B * T);
        a.fc1 = dfloat((size_t)B * T * Hd); a.gact = dfloat((size_t)B * T * Hd);
        a.fc2 = dfloat(BTD); a.res2 = dfloat(BTD);

        a.dln1_out = dfloat(BTD); a.dqkv = dfloat((size_t)BTD * 3);
        a.dq = dfloat(BTD); a.dk = dfloat(BTD); a.dv = dfloat(BTD);
        a.dscores = dfloat(BHTT); a.dprobs = dfloat(BHTT);
        a.dattn = dfloat(BTD); a.dmerged = dfloat(BTD); a.dproj = dfloat(BTD); a.dres1 = dfloat(BTD);
        a.dln2_out = dfloat(BTD); a.dfc1 = dfloat((size_t)B * T * Hd); a.dgact = dfloat((size_t)B * T * Hd);
        a.dfc2 = dfloat(BTD); a.dx_out = dfloat(BTD);
        a.x_in = nullptr;   /* se fija en el forward */
    }

    m.cls_feat = dfloat((size_t)B * D);
    m.lnf_out  = dfloat((size_t)B * D);
    m.lnf_mean = dfloat(B); m.lnf_rstd = dfloat(B);
    m.logits = dfloat((size_t)B * C); m.probs_out = dfloat((size_t)B * C); m.loss = dfloat(B);
    m.dlogits = dfloat((size_t)B * C); m.dlnf_out = dfloat((size_t)B * D);
    m.dcls_feat = dfloat((size_t)B * D);
    m.dtokens = dfloat(BTD);
    m.demb    = dfloat((size_t)B * NP * D);
    CUDA_CHECK(cudaMalloc(&m.d_correct, sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&m.h_loss, (size_t)B * sizeof(float)));

    m.adam_t = 0;
}

void vit_free(ViT &m)
{
    if (m.params) { cudaFree(m.params); cudaFree(m.grads); cudaFree(m.adam_m); cudaFree(m.adam_v); }
    if (m.blocks) {
        for (int l = 0; l < m.cfg.n_blocks; ++l) {
            BlockAct &a = m.blocks[l];
            cudaFree(a.ln1_out); cudaFree(a.ln1_mean); cudaFree(a.ln1_rstd);
            cudaFree(a.qkv); cudaFree(a.q); cudaFree(a.k); cudaFree(a.v);
            cudaFree(a.scores); cudaFree(a.probs); cudaFree(a.attn); cudaFree(a.merged);
            cudaFree(a.proj); cudaFree(a.res1);
            cudaFree(a.ln2_out); cudaFree(a.ln2_mean); cudaFree(a.ln2_rstd);
            cudaFree(a.fc1); cudaFree(a.gact); cudaFree(a.fc2); cudaFree(a.res2);
            cudaFree(a.dln1_out); cudaFree(a.dqkv); cudaFree(a.dq); cudaFree(a.dk); cudaFree(a.dv);
            cudaFree(a.dscores); cudaFree(a.dprobs); cudaFree(a.dattn); cudaFree(a.dmerged);
            cudaFree(a.dproj); cudaFree(a.dres1);
            cudaFree(a.dln2_out); cudaFree(a.dfc1); cudaFree(a.dgact); cudaFree(a.dfc2); cudaFree(a.dx_out);
        }
        delete[] m.blocks;
    }
    cudaFree(m.d_images); cudaFree(m.d_labels);
    cudaFree(m.patches); cudaFree(m.emb); cudaFree(m.tokens);
    cudaFree(m.cls_feat); cudaFree(m.lnf_out); cudaFree(m.lnf_mean); cudaFree(m.lnf_rstd);
    cudaFree(m.logits); cudaFree(m.probs_out); cudaFree(m.loss);
    cudaFree(m.dlogits); cudaFree(m.dlnf_out); cudaFree(m.dcls_feat);
    cudaFree(m.dtokens); cudaFree(m.demb); cudaFree(m.d_correct);
    if (m.h_loss) cudaFreeHost(m.h_loss);

    delete[] m.p.ln1_g; delete[] m.p.ln1_b; delete[] m.p.Wqkv; delete[] m.p.bqkv;
    delete[] m.p.Wproj; delete[] m.p.bproj; delete[] m.p.ln2_g; delete[] m.p.ln2_b;
    delete[] m.p.Wfc1;  delete[] m.p.bfc1;  delete[] m.p.Wfc2;  delete[] m.p.bfc2;
    delete[] m.g.ln1_g; delete[] m.g.ln1_b; delete[] m.g.Wqkv; delete[] m.g.bqkv;
    delete[] m.g.Wproj; delete[] m.g.bproj; delete[] m.g.ln2_g; delete[] m.g.ln2_b;
    delete[] m.g.Wfc1;  delete[] m.g.bfc1;  delete[] m.g.Wfc2;  delete[] m.g.bfc2;
    /* ViT tiene miembros con inicializadores por defecto (ViTConfig), lo que
     * lo vuelve un tipo no trivial: memset sobre el es comportamiento
     * indefinido segun el estandar de C++. La reasignacion de un objeto
     * recien construido logra el mismo "poner todo en cero/valores por
     * defecto" sin ese problema. */
    m = ViT{};
}

/* ==========================================================================
 *  FORWARD
 * ========================================================================*/
float vit_forward(ViT &m, const float *d_batch_images, const int *d_batch_labels,
                  int n_valid, bool want_loss)
{
    const ViTConfig &c = m.cfg;
    const int B  = n_valid;              /* el batch efectivo puede ser menor */
    const int T  = c.n_tokens();
    const int D  = c.d_model;
    const int H  = c.n_heads;
    const int Dh = c.head_dim();
    const int Hd = c.mlp_hidden;
    const int NP = c.n_patches();
    const int PD = c.patch_dim();
    const int C  = c.n_classes;
    const float scale = 1.0f / std::sqrt((float)Dh);

    m.cur_labels = d_batch_labels;

    /* 1. imagen -> parches -> proyeccion lineal */
    im2patch(d_batch_images, m.patches, B, c.img_size, c.patch);
    gemm_nn(m.patches, m.p.W_patch, m.p.b_patch, m.emb, B * NP, D, PD);

    /* 2. anteponer [CLS] y sumar embeddings posicionales */
    assemble_tokens(m.emb, m.p.cls, m.p.pos, m.tokens, B, NP, D);

    float *x = m.tokens;

    /* 3. bloques Transformer (pre-norm) */
    for (int l = 0; l < c.n_blocks; ++l) {
        BlockAct &a = m.blocks[l];
        a.x_in = x;

        /* --- sub-bloque de atencion --- */
        layernorm_forward(x, m.p.ln1_g[l], m.p.ln1_b[l],
                          a.ln1_out, a.ln1_mean, a.ln1_rstd, B * T, D, c.ln_eps);

        /* proyeccion conjunta a Q, K y V en un solo GEMM: reduce el numero de
         * lanzamientos y mejora la reutilizacion de la entrada. */
        gemm_nn(a.ln1_out, m.p.Wqkv[l], m.p.bqkv[l], a.qkv, B * T, 3 * D, D);
        split_qkv(a.qkv, a.q, a.k, a.v, B, T, H, Dh);

        /* scores = Q*K^T / sqrt(Dh)   [B*H, T, T]   <-- KERNEL DEL EXPERIMENTO 2 */
        bmm_nt(a.q, a.k, a.scores, B * H, T, T, Dh,
               T * Dh, T * Dh, T * T, scale, c.attn_mem);

        softmax_rows(a.scores, a.probs, B * H * T, T);

        /* attn = P*V                  [B*H, T, Dh]  <-- KERNEL DEL EXPERIMENTO 2 */
        bmm_nn(a.probs, a.v, a.attn, B * H, T, Dh, T,
               T * T, T * Dh, T * Dh, 1.0f, c.attn_mem);

        merge_heads(a.attn, a.merged, B, T, H, Dh);
        gemm_nn(a.merged, m.p.Wproj[l], m.p.bproj[l], a.proj, B * T, D, D);
        add_forward(x, a.proj, a.res1, B * T * D);          /* residual 1 */

        /* --- sub-bloque MLP --- */
        layernorm_forward(a.res1, m.p.ln2_g[l], m.p.ln2_b[l],
                          a.ln2_out, a.ln2_mean, a.ln2_rstd, B * T, D, c.ln_eps);
        gemm_nn(a.ln2_out, m.p.Wfc1[l], m.p.bfc1[l], a.fc1, B * T, Hd, D);
        gelu_forward(a.fc1, a.gact, B * T * Hd);
        gemm_nn(a.gact, m.p.Wfc2[l], m.p.bfc2[l], a.fc2, B * T, D, Hd);
        add_forward(a.res1, a.fc2, a.res2, B * T * D);      /* residual 2 */

        x = a.res2;
    }

    /* 4. cabeza de clasificacion sobre el token [CLS] */
    {
        const int n = B * D, thr = 256;
        extract_cls_kernel<<<ceil_div(n, thr), thr>>>(x, m.cls_feat, B, T, D);
        CUDA_CHECK_KERNEL();
    }
    layernorm_forward(m.cls_feat, m.p.lnf_g, m.p.lnf_b,
                      m.lnf_out, m.lnf_mean, m.lnf_rstd, B, D, c.ln_eps);
    gemm_nn(m.lnf_out, m.p.Whead, m.p.bhead, m.logits, B, C, D);

    if (!want_loss || d_batch_labels == nullptr) return 0.0f;

    softmax_cross_entropy_forward(m.logits, d_batch_labels, m.probs_out, m.loss, B, C);

    /* La media sobre el batch se hace en host: son B floats, la transferencia
     * es despreciable y evita un kernel de reduccion adicional. */
    CUDA_CHECK(cudaMemcpy(m.h_loss, m.loss, (size_t)B * sizeof(float), cudaMemcpyDeviceToHost));
    double s = 0.0;
    for (int i = 0; i < B; ++i) s += m.h_loss[i];
    return (float)(s / B);
}

/* ==========================================================================
 *  BACKWARD  (todas las derivadas son analiticas y estan anotadas)
 * ========================================================================*/
void vit_backward(ViT &m, int n_valid)
{
    const ViTConfig &c = m.cfg;
    const int B  = n_valid;
    const int T  = c.n_tokens();
    const int D  = c.d_model;
    const int H  = c.n_heads;
    const int Dh = c.head_dim();
    const int Hd = c.mlp_hidden;
    const int NP = c.n_patches();
    const int PD = c.patch_dim();
    const int C  = c.n_classes;
    const float scale = 1.0f / std::sqrt((float)Dh);

    /* --- perdida: dlogits = (softmax(logits) - onehot)/B --- */
    softmax_cross_entropy_backward(m.probs_out, m.cur_labels, m.dlogits, B, C);

    /* --- cabeza lineal: logits = lnf_out * Whead + bhead --- */
    gemm_nt(m.dlogits, m.p.Whead, m.dlnf_out, B, D, C);          /* dX = dY*W^T */
    gemm_tn(m.lnf_out, m.dlogits, m.g.Whead, D, C, B);           /* dW = X^T*dY */
    bias_backward(m.dlogits, m.g.bhead, B, C);

    /* --- LayerNorm final --- */
    layernorm_backward(m.dlnf_out, m.cls_feat, m.p.lnf_g, m.lnf_mean, m.lnf_rstd,
                       m.dcls_feat, m.g.lnf_g, m.g.lnf_b, B, D);

    /* --- el gradiente vuelve a la secuencia solo por la posicion del [CLS] --- */
    {
        const int n = B * T * D, thr = 256;
        scatter_cls_grad_kernel<<<ceil_div(n, thr), thr>>>(m.dcls_feat, m.dtokens, B, T, D);
        CUDA_CHECK_KERNEL();
    }

    float *dx = m.dtokens;   /* gradiente respecto de la salida del ultimo bloque */

    for (int l = c.n_blocks - 1; l >= 0; --l) {
        BlockAct &a = m.blocks[l];

        /* ---- residual 2: res2 = res1 + fc2  =>  dres1 = dx, dfc2 = dx ---- */
        CUDA_CHECK(cudaMemcpy(a.dfc2,  dx, (size_t)B * T * D * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(a.dres1, dx, (size_t)B * T * D * sizeof(float), cudaMemcpyDeviceToDevice));

        /* ---- fc2: fc2 = gact * Wfc2 + bfc2 ---- */
        gemm_nt(a.dfc2, m.p.Wfc2[l], a.dgact, B * T, Hd, D);
        gemm_tn(a.gact, a.dfc2, m.g.Wfc2[l], Hd, D, B * T);
        bias_backward(a.dfc2, m.g.bfc2[l], B * T, D);

        /* ---- GELU ---- */
        gelu_backward(a.fc1, a.dgact, a.dfc1, B * T * Hd);

        /* ---- fc1: fc1 = ln2_out * Wfc1 + bfc1 ---- */
        gemm_nt(a.dfc1, m.p.Wfc1[l], a.dln2_out, B * T, D, Hd);
        gemm_tn(a.ln2_out, a.dfc1, m.g.Wfc1[l], D, Hd, B * T);
        bias_backward(a.dfc1, m.g.bfc1[l], B * T, Hd);

        /* ---- LayerNorm 2. Su dx se SUMA al que ya venia por el atajo
         *      residual (res1 alimenta tanto a la rama MLP como a la suma). */
        layernorm_backward(a.dln2_out, a.res1, m.p.ln2_g[l], a.ln2_mean, a.ln2_rstd,
                           a.dproj, m.g.ln2_g[l], m.g.ln2_b[l], B * T, D);
        add_inplace(a.dres1, a.dproj, B * T * D);
        /* a partir de aqui a.dres1 es el gradiente total respecto de res1 */

        /* ---- residual 1: res1 = x_in + proj  =>  dproj = dres1, dx_in += dres1 ---- */
        CUDA_CHECK(cudaMemcpy(a.dproj, a.dres1, (size_t)B * T * D * sizeof(float), cudaMemcpyDeviceToDevice));

        /* ---- proyeccion de salida de la atencion ---- */
        gemm_nt(a.dproj, m.p.Wproj[l], a.dmerged, B * T, D, D);
        gemm_tn(a.merged, a.dproj, m.g.Wproj[l], D, D, B * T);
        bias_backward(a.dproj, m.g.bproj[l], B * T, D);

        split_heads(a.dmerged, a.dattn, B, T, H, Dh);

        /* ---- attn = P*V ----
         *   dP = dAttn * V^T        [T,T]
         *   dV = P^T * dAttn        [T,Dh]                                   */
        bmm_nt(a.dattn, a.v, a.dprobs, B * H, T, T, Dh,
               T * Dh, T * Dh, T * T, 1.0f, c.attn_mem);
        bmm_tn(a.probs, a.dattn, a.dv, B * H, T, Dh, T,
               T * T, T * Dh, T * Dh, 1.0f, c.attn_mem);

        /* ---- softmax por filas. El factor `scale` convierte el gradiente
         *      respecto de los scores escalados en gradiente respecto de Q*K^T. */
        softmax_backward(a.probs, a.dprobs, a.dscores, B * H * T, T, scale);

        /* ---- scores = Q*K^T ----
         *   dQ = dS * K             [T,Dh]
         *   dK = dS^T * Q           [T,Dh]                                   */
        bmm_nn(a.dscores, a.k, a.dq, B * H, T, Dh, T,
               T * T, T * Dh, T * Dh, 1.0f, c.attn_mem);
        bmm_tn(a.dscores, a.q, a.dk, B * H, T, Dh, T,
               T * T, T * Dh, T * Dh, 1.0f, c.attn_mem);

        merge_qkv_grad(a.dq, a.dk, a.dv, a.dqkv, B, T, H, Dh);

        /* ---- proyeccion QKV ---- */
        gemm_nt(a.dqkv, m.p.Wqkv[l], a.dln1_out, B * T, D, 3 * D);
        gemm_tn(a.ln1_out, a.dqkv, m.g.Wqkv[l], D, 3 * D, B * T);
        bias_backward(a.dqkv, m.g.bqkv[l], B * T, 3 * D);

        /* ---- LayerNorm 1: su dx se suma al atajo residual ---- */
        layernorm_backward(a.dln1_out, a.x_in, m.p.ln1_g[l], a.ln1_mean, a.ln1_rstd,
                           a.dx_out, m.g.ln1_g[l], m.g.ln1_b[l], B * T, D);
        add_inplace(a.dx_out, a.dres1, B * T * D);

        dx = a.dx_out;   /* gradiente respecto de la entrada de este bloque */
    }

    /* --- embeddings: [CLS], posicionales y proyeccion de parches --- */
    assemble_tokens_backward(dx, m.demb, m.g.cls, m.g.pos, B, NP, D);
    gemm_tn(m.patches, m.demb, m.g.W_patch, PD, D, B * NP);
    bias_backward(m.demb, m.g.b_patch, B * NP, D);
}

void vit_zero_grad(ViT &m)
{
    CUDA_CHECK(cudaMemset(m.grads, 0, (size_t)m.n_params * sizeof(float)));
}

void vit_adam(ViT &m, float lr, float weight_decay)
{
    m.adam_t += 1;
    adam_step(m.params, m.grads, m.adam_m, m.adam_v, m.n_params,
              lr, 0.9f, 0.999f, 1e-8f, weight_decay, m.adam_t);
}
