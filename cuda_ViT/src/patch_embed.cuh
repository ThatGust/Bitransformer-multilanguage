/* ============================================================================
 * patch_embed.cuh -- Conversion de imagen a secuencia de tokens.
 *
 * La imagen 28x28 se corta en parches cuadrados NO solapados de patch x patch.
 * Con patch = 4, 7 y 14 salen 7x7=49, 4x4=16 y 2x2=4 parches respectivamente
 * (28 es divisible por los tres, por eso se eligieron).
 *
 * Cada parche se aplana a un vector de patch*patch valores y se proyecta
 * linealmente a dimension D. Luego se antepone el token [CLS] aprendible y se
 * suma el embedding posicional, dando la secuencia de T = n_patches+1 tokens
 * que consumen los bloques Transformer.
 * ==========================================================================*/
#ifndef PATCH_EMBED_CUH
#define PATCH_EMBED_CUH

/* images[B, img*img] -> patches[B*n_patches, patch_dim] (im2col sin solape). */
void im2patch(const float *images, float *patches,
              int B, int img_size, int patch, cudaStream_t stream = 0);

/* tokens[b,0,:]   = cls[:] + pos[0,:]
 * tokens[b,1+p,:] = emb[b,p,:] + pos[1+p,:]                                  */
void assemble_tokens(const float *emb, const float *cls, const float *pos,
                     float *tokens, int B, int n_patches, int D,
                     cudaStream_t stream = 0);

/* Backward de assemble_tokens: reparte dtokens entre la proyeccion de parches,
 * el token CLS y los embeddings posicionales. Como cls y pos se comparten
 * entre todas las muestras del batch, sus gradientes son sumas sobre b. */
void assemble_tokens_backward(const float *dtokens, float *demb,
                              float *dcls, float *dpos,
                              int B, int n_patches, int D,
                              cudaStream_t stream = 0);

#endif /* PATCH_EMBED_CUH */
