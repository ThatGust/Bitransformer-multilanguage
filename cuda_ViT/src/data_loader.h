/* ============================================================================
 * data_loader.h -- Lectura del dataset MNIST en formato IDX (archivos .ubyte)
 *
 * El formato IDX guarda los enteros del header en big-endian, mientras que
 * x86-64 y ARM (Colab y Mac) son little-endian, por lo que hay que invertir
 * los bytes al leer.
 *
 *   IDX3 (imagenes):  [0..3] magic = 0x00000803
 *                     [4..7] n_images
 *                     [8..11] n_rows
 *                     [12..15] n_cols
 *                     [16..]  n_images*n_rows*n_cols bytes uint8
 *
 *   IDX1 (etiquetas): [0..3] magic = 0x00000801
 *                     [4..7] n_labels
 *                     [8..]  n_labels bytes uint8
 *
 * Las imagenes se normalizan a float en [0,1] y se guardan en un unico bloque
 * contiguo, listo para un solo cudaMemcpy host->device.
 * ==========================================================================*/
#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#ifdef __cplusplus
extern "C" {
#endif

/* Un split (train o test) completamente cargado en memoria de host.
 * images: bloque contiguo de n_samples*rows*cols floats en [0,1],
 *         ordenado como images[i*rows*cols + r*cols + c].
 * labels: n_samples enteros en [0,9].
 * El layout contiguo es intencional: permite copiar todo el split a GPU con
 * una sola llamada a cudaMemcpy sin reempaquetar nada. */
typedef struct {
    float *images;
    int   *labels;
    int    n_samples;
    int    rows;
    int    cols;
} MnistData;

/* Carga un split desde los dos archivos IDX. Si max_samples > 0 solo se cargan
 * los primeros max_samples ejemplos (esto es lo que permite usar el subconjunto
 * reducido de 5000/2000 sin tocar el codigo, ver --train-n / --test-n).
 * Devuelve 0 en exito, distinto de 0 en error (con mensaje en stderr). */
int mnist_load(const char *images_path,
               const char *labels_path,
               int max_samples,
               MnistData *out);

/* Libera los buffers de host. Seguro de llamar sobre una estructura en cero. */
void mnist_free(MnistData *d);

/* Imprime una imagen como ASCII art junto con su etiqueta. Es el sanity check
 * que se corre ANTES de entrenar: si el digito no se ve, el parser del header
 * o el stride estan mal y no tiene sentido seguir. */
void mnist_print_ascii(const MnistData *d, int index);

#ifdef __cplusplus
}
#endif

#endif /* DATA_LOADER_H */
