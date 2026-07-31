/* ============================================================================
 * ops.cuh -- Operaciones elementales: GELU, residuales, perdida y optimizador.
 * ==========================================================================*/
#ifndef OPS_CUH
#define OPS_CUH

/* GELU (aproximacion tanh de Hendrycks & Gimpel). Se elige GELU sobre ReLU
 * porque es la activacion estandar de los Transformers y, al ser suave, no
 * tiene el problema de las unidades muertas de ReLU con un modelo tan pequeno. */
void gelu_forward(const float *x, float *y, int n, cudaStream_t stream = 0);
void gelu_backward(const float *x, const float *dy, float *dx, int n, cudaStream_t stream = 0);

/* out = a + b (conexion residual). El backward de una suma es la identidad:
 * el gradiente se copia a ambas ramas, por eso no hace falta kernel inverso,
 * solo acumular con add_inplace. */
void add_forward(const float *a, const float *b, float *out, int n, cudaStream_t stream = 0);
void add_inplace(float *a, const float *b, int n, cudaStream_t stream = 0);

void fill_zero(float *x, int n, cudaStream_t stream = 0);

/* Softmax + entropia cruzada sobre logits[B,C].
 * Devuelve las probabilidades y la perdida por muestra (loss_per_sample[B]);
 * la media sobre el batch se hace en host para evitar una reduccion extra. */
void softmax_cross_entropy_forward(const float *logits, const int *labels,
                                   float *probs, float *loss_per_sample,
                                   int B, int C, cudaStream_t stream = 0);

/* dlogits = (probs - onehot(label)) / B.
 * Es el gradiente cerrado de softmax+CE combinados: la jacobiana del softmax y
 * la derivada del log se cancelan y queda esta expresion, mucho mas estable
 * numericamente que encadenar ambas por separado. */
void softmax_cross_entropy_backward(const float *probs, const int *labels,
                                    float *dlogits, int B, int C,
                                    cudaStream_t stream = 0);

/* Numero de aciertos del batch (argmax de probs vs label), acumulado en
 * d_correct con atomicAdd. */
void count_correct(const float *probs, const int *labels, int *d_correct,
                   int B, int C, cudaStream_t stream = 0);

/* Adam con correccion de sesgo:
 *   m = b1*m + (1-b1)*g ;  v = b2*v + (1-b2)*g^2
 *   mhat = m/(1-b1^t) ;    vhat = v/(1-b2^t)
 *   p -= lr * mhat / (sqrt(vhat)+eps)  [+ weight decay desacoplado]
 * Se usa Adam y no SGD porque los Transformers son muy sensibles al learning
 * rate con SGD y con solo 10-15 epocas no habria margen para ajustarlo. */
void adam_step(float *param, const float *grad, float *m, float *v,
               int n, float lr, float beta1, float beta2, float eps,
               float weight_decay, int t, cudaStream_t stream = 0);

#endif /* OPS_CUH */
