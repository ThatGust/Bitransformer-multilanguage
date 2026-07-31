/* ============================================================================
 * data_loader.c -- Implementacion del lector IDX de MNIST.
 * Compilable como C puro (gcc/clang) o como parte de una unidad CUDA.
 * No depende de CUDA: solo produce buffers de host contiguos.
 * ==========================================================================*/
#include "data_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Lee 4 bytes big-endian y los convierte al orden nativo.
 * Se hace byte a byte en vez de con __builtin_bswap32 sobre un uint32_t leido
 * directamente para no depender del endianness de la maquina: este codigo da
 * el mismo resultado en little-endian y en big-endian. */
static int read_be_u32(FILE *f, unsigned int *out)
{
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return -1;
    *out = ((unsigned int)b[0] << 24) |
           ((unsigned int)b[1] << 16) |
           ((unsigned int)b[2] << 8)  |
           ((unsigned int)b[3]);
    return 0;
}

int mnist_load(const char *images_path,
               const char *labels_path,
               int max_samples,
               MnistData *out)
{
    FILE *fi = NULL, *fl = NULL;
    unsigned int magic, n_img, n_rows, n_cols, n_lab;
    unsigned char *raw_px = NULL, *raw_lb = NULL;
    long n, npix, i;

    memset(out, 0, sizeof(*out));

    fi = fopen(images_path, "rb");
    if (!fi) { fprintf(stderr, "[data_loader] no se pudo abrir %s\n", images_path); goto fail; }
    fl = fopen(labels_path, "rb");
    if (!fl) { fprintf(stderr, "[data_loader] no se pudo abrir %s\n", labels_path); goto fail; }

    /* --- Header del archivo de imagenes (IDX3) --- */
    if (read_be_u32(fi, &magic) || read_be_u32(fi, &n_img) ||
        read_be_u32(fi, &n_rows) || read_be_u32(fi, &n_cols)) {
        fprintf(stderr, "[data_loader] header de imagenes truncado en %s\n", images_path);
        goto fail;
    }
    if (magic != 0x00000803u) {
        fprintf(stderr, "[data_loader] magic de imagenes invalido: 0x%08x (esperado 0x00000803)\n", magic);
        goto fail;
    }

    /* --- Header del archivo de etiquetas (IDX1) --- */
    if (read_be_u32(fl, &magic) || read_be_u32(fl, &n_lab)) {
        fprintf(stderr, "[data_loader] header de etiquetas truncado en %s\n", labels_path);
        goto fail;
    }
    if (magic != 0x00000801u) {
        fprintf(stderr, "[data_loader] magic de etiquetas invalido: 0x%08x (esperado 0x00000801)\n", magic);
        goto fail;
    }
    if (n_lab != n_img) {
        fprintf(stderr, "[data_loader] desajuste imagenes(%u) vs etiquetas(%u)\n", n_img, n_lab);
        goto fail;
    }

    n = (long)n_img;
    if (max_samples > 0 && max_samples < n) n = max_samples;
    npix = (long)n_rows * (long)n_cols;

    raw_px = (unsigned char *)malloc((size_t)n * (size_t)npix);
    raw_lb = (unsigned char *)malloc((size_t)n);
    out->images = (float *)malloc((size_t)n * (size_t)npix * sizeof(float));
    out->labels = (int *)malloc((size_t)n * sizeof(int));
    if (!raw_px || !raw_lb || !out->images || !out->labels) {
        fprintf(stderr, "[data_loader] sin memoria para %ld muestras\n", n);
        goto fail;
    }

    /* Lectura en bloque: los primeros n ejemplos son contiguos en el archivo,
     * asi que el subconjunto reducido no requiere seeks. */
    if (fread(raw_px, 1, (size_t)n * (size_t)npix, fi) != (size_t)n * (size_t)npix) {
        fprintf(stderr, "[data_loader] lectura de pixeles incompleta\n"); goto fail;
    }
    if (fread(raw_lb, 1, (size_t)n, fl) != (size_t)n) {
        fprintf(stderr, "[data_loader] lectura de etiquetas incompleta\n"); goto fail;
    }

    /* Normalizacion a [0,1]. Se usa la escala simple x/255 en vez de
     * estandarizar con media/desvio de MNIST porque el modelo lleva LayerNorm
     * inmediatamente despues del patch embedding, que reabsorbe cualquier
     * corrimiento de escala global. */
    for (i = 0; i < n * npix; ++i) out->images[i] = (float)raw_px[i] * (1.0f / 255.0f);
    for (i = 0; i < n; ++i)        out->labels[i] = (int)raw_lb[i];

    out->n_samples = (int)n;
    out->rows = (int)n_rows;
    out->cols = (int)n_cols;

    free(raw_px); free(raw_lb);
    fclose(fi); fclose(fl);
    return 0;

fail:
    if (raw_px) free(raw_px);
    if (raw_lb) free(raw_lb);
    if (fi) fclose(fi);
    if (fl) fclose(fl);
    mnist_free(out);
    return -1;
}

void mnist_free(MnistData *d)
{
    if (!d) return;
    if (d->images) free(d->images);
    if (d->labels) free(d->labels);
    memset(d, 0, sizeof(*d));
}

void mnist_print_ascii(const MnistData *d, int index)
{
    static const char ramp[] = " .:-=+*#%@";   /* 10 niveles de intensidad */
    int r, c;
    const float *img;

    if (!d || !d->images || index < 0 || index >= d->n_samples) {
        fprintf(stderr, "[data_loader] indice %d fuera de rango\n", index);
        return;
    }
    img = d->images + (size_t)index * d->rows * d->cols;

    printf("--- muestra %d | etiqueta = %d | %dx%d ---\n",
           index, d->labels[index], d->rows, d->cols);
    for (r = 0; r < d->rows; ++r) {
        for (c = 0; c < d->cols; ++c) {
            float v = img[r * d->cols + c];
            int level = (int)(v * 9.0f + 0.5f);
            if (level < 0) level = 0;
            if (level > 9) level = 9;
            /* dos caracteres por pixel para compensar la relacion de aspecto
             * de la terminal y que el digito no salga aplastado */
            putchar(ramp[level]); putchar(ramp[level]);
        }
        putchar('\n');
    }
}

/* ---------------------------------------------------------------------------
 * Test standalone. Compilar y correr ANTES de entrenar:
 *     gcc -DDATA_LOADER_TEST -O2 -o test_loader data_loader.c && ./test_loader
 * Verifica los 4 archivos, imprime dimensiones y dibuja algunos digitos.
 * -------------------------------------------------------------------------*/
#ifdef DATA_LOADER_TEST
int main(int argc, char **argv)
{
    const char *dir = (argc > 1) ? argv[1] : "../data";
    char p_tri[512], p_trl[512], p_tei[512], p_tel[512];
    MnistData train, test;
    int i, k;
    double sum = 0.0;
    int hist[10];

    snprintf(p_tri, sizeof(p_tri), "%s/train-images-idx3-ubyte", dir);
    snprintf(p_trl, sizeof(p_trl), "%s/train-labels-idx1-ubyte", dir);
    snprintf(p_tei, sizeof(p_tei), "%s/t10k-images-idx3-ubyte", dir);
    snprintf(p_tel, sizeof(p_tel), "%s/t10k-labels-idx1-ubyte", dir);

    if (mnist_load(p_tri, p_trl, 0, &train)) return 1;
    if (mnist_load(p_tei, p_tel, 0, &test))  return 1;

    printf("train: %d muestras de %dx%d\n", train.n_samples, train.rows, train.cols);
    printf("test : %d muestras de %dx%d\n", test.n_samples, test.rows, test.cols);

    /* Chequeos numericos: el rango debe ser [0,1] y la media de MNIST
     * normalizado es ~0.1307, un valor conocido que delata errores de stride. */
    for (i = 0; i < train.n_samples * train.rows * train.cols; ++i) {
        float v = train.images[i];
        if (v < 0.0f || v > 1.0f) { printf("FALLO: pixel fuera de [0,1]: %f\n", v); return 1; }
        sum += v;
    }
    printf("media de pixeles = %.4f (referencia MNIST ~0.1307)\n",
           sum / ((double)train.n_samples * train.rows * train.cols));

    memset(hist, 0, sizeof(hist));
    for (i = 0; i < train.n_samples; ++i) {
        if (train.labels[i] < 0 || train.labels[i] > 9) { printf("FALLO: etiqueta %d\n", train.labels[i]); return 1; }
        hist[train.labels[i]]++;
    }
    printf("distribucion de clases en train:\n");
    for (k = 0; k < 10; ++k) printf("  digito %d: %6d\n", k, hist[k]);

    for (i = 0; i < 3; ++i) mnist_print_ascii(&train, i);
    mnist_print_ascii(&test, 0);

    mnist_free(&train);
    mnist_free(&test);
    printf("\n[OK] el loader funciona, los datos se pueden usar para entrenar.\n");
    return 0;
}
#endif
