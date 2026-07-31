/* ============================================================================
 * vit.cuh -- Ensamblado del Vision Transformer.
 *
 * Arquitectura (de entrada a salida):
 *
 *   imagen 28x28
 *     -> im2patch            [B, n_patches, patch_dim]
 *     -> proyeccion lineal   [B, n_patches, D]
 *     -> + [CLS] + pos       [B, T, D]      con T = n_patches + 1
 *     -> n_blocks x {
 *            x = x + Attn(LN(x))            (pre-norm)
 *            x = x + MLP(LN(x))
 *        }
 *     -> token CLS -> LN final -> lineal -> 10 logits -> softmax
 *
 * Se usa la variante PRE-NORM (LayerNorm antes de cada submodulo, no despues)
 * porque estabiliza el entrenamiento sin necesidad de warmup del learning
 * rate, algo importante cuando solo se dispone de 10-15 epocas.
 *
 * Todos los parametros viven en UN unico bloque contiguo de memoria de device
 * (params), con sus gradientes y los estados de Adam en bloques paralelos de
 * identico tamano. Asi el paso del optimizador es un unico lanzamiento de
 * kernel sobre todo el modelo en vez de uno por tensor.
 * ==========================================================================*/
#ifndef VIT_CUH
#define VIT_CUH

#include "attention.cuh"

struct ViTConfig {
    int img_size   = 28;
    int patch      = 7;    /* 4, 7 o 14 -> 49, 16 o 4 parches */
    int d_model    = 64;
    int n_heads    = 4;
    int n_blocks   = 2;
    int mlp_hidden = 128;  /* 2 * d_model */
    int n_classes  = 10;
    int batch      = 64;
    float ln_eps   = 1e-5f;
    AttnMem attn_mem = ATTN_SHARED;

    /* derivados */
    int patch_dim()  const { return patch * patch; }
    int grid_side()  const { return img_size / patch; }
    int n_patches()  const { return grid_side() * grid_side(); }
    int n_tokens()   const { return n_patches() + 1; }
    int head_dim()   const { return d_model / n_heads; }
};

/* Punteros a los parametros dentro del bloque contiguo. Los gradientes usan
 * exactamente el mismo desplazamiento dentro del bloque de gradientes. */
struct ViTParams {
    float *W_patch, *b_patch;
    float *cls, *pos;
    /* por bloque (arreglos de tamano n_blocks) */
    float **ln1_g, **ln1_b;
    float **Wqkv,  **bqkv;
    float **Wproj, **bproj;
    float **ln2_g, **ln2_b;
    float **Wfc1,  **bfc1;
    float **Wfc2,  **bfc2;
    float *lnf_g, *lnf_b;
    float *Whead, *bhead;
};

/* Activaciones intermedias de un bloque Transformer. Se guardan todas porque
 * el backward analitico las necesita (no se recomputa nada). */
struct BlockAct {
    float *x_in;                              /* entrada del bloque [B,T,D]  */
    float *ln1_out, *ln1_mean, *ln1_rstd;
    float *qkv, *q, *k, *v;
    float *scores, *probs, *attn, *merged, *proj, *res1;
    float *ln2_out, *ln2_mean, *ln2_rstd;
    float *fc1, *gact, *fc2, *res2;
    /* gradientes espejo */
    float *dln1_out, *dqkv, *dq, *dk, *dv;
    float *dscores, *dprobs, *dattn, *dmerged, *dproj, *dres1;
    float *dln2_out, *dfc1, *dgact, *dfc2, *dx_out;
};

struct ViT {
    ViTConfig cfg;

    /* parametros y estado del optimizador (bloques contiguos) */
    float *params = nullptr, *grads = nullptr, *adam_m = nullptr, *adam_v = nullptr;
    int    n_params = 0;
    ViTParams p{}, g{};

    /* activaciones */
    float *d_images = nullptr;      /* [B, img*img]        */
    int   *d_labels = nullptr;      /* [B]                 */
    float *patches = nullptr, *emb = nullptr, *tokens = nullptr;
    BlockAct *blocks = nullptr;
    float *cls_feat = nullptr, *lnf_out = nullptr, *lnf_mean = nullptr, *lnf_rstd = nullptr;
    float *logits = nullptr, *probs_out = nullptr, *loss = nullptr;
    float *dlogits = nullptr, *dlnf_out = nullptr, *dcls_feat = nullptr, *dtokens = nullptr;
    float *demb = nullptr;
    int   *d_correct = nullptr;
    float *h_loss = nullptr;        /* buffer de host (pinned) para la perdida */

    /* Etiquetas del batch en curso. vit_forward las guarda aqui para que
     * vit_backward pueda calcular dlogits sin volver a recibirlas. */
    const int *cur_labels = nullptr;

    int adam_t = 0;                 /* contador de pasos, para el bias correction */
};

void vit_init(ViT &m, const ViTConfig &cfg, unsigned long long seed);
void vit_free(ViT &m);

/* Forward completo. Si labels != nullptr calcula tambien la perdida.
 * Devuelve la perdida media del batch (0 si labels == nullptr).
 * n_valid permite procesar un ultimo batch incompleto. */
float vit_forward(ViT &m, const float *d_batch_images, const int *d_batch_labels,
                  int n_valid, bool want_loss);

/* Backward completo. Debe llamarse justo despues de vit_forward. */
void vit_backward(ViT &m, int n_valid);

/* Un paso de Adam sobre todos los parametros. */
void vit_adam(ViT &m, float lr, float weight_decay);

/* Pone a cero el bloque de gradientes. */
void vit_zero_grad(ViT &m);

#endif /* VIT_CUH */
