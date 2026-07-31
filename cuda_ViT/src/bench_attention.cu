/* ============================================================================
 * bench_attention.cu -- EXPERIMENTO 2 aislado: memoria global vs compartida.
 *
 * Mide, con cudaEvent_t, el tiempo de los DOS kernels criticos de la atencion
 * por separado del resto del entrenamiento:
 *
 *   QK^T  : [T,Dh] x [T,Dh]^T -> [T,T]     (bmm_nt)
 *   P*V   : [T,T]  x [T,Dh]   -> [T,Dh]    (bmm_nn)
 *
 * para las tres longitudes de secuencia que produce el experimento 1
 * (T = 50, 17 y 5 tokens, correspondientes a parches de 4x4, 7x7 y 14x14).
 *
 * Metodologia de medicion:
 *   - Se comprueba primero que ambas versiones dan el MISMO resultado
 *     numerico (tolerancia de punto flotante), porque un speedup sobre un
 *     kernel incorrecto no significa nada.
 *   - Iteraciones de calentamiento descartadas: la primera invocacion de un
 *     kernel incluye la compilacion JIT del PTX y la carga del modulo.
 *   - Se repite el kernel n_iters veces dentro de una unica ventana de
 *     medicion y se divide, para que la latencia de lanzamiento (~5-10 us) no
 *     domine sobre kernels que duran decenas de microsegundos.
 *   - Se reporta la mediana de n_repeats ventanas para descartar valores
 *     atipicos por interferencia de otros procesos en la GPU compartida.
 *
 * Salida: results/bench_attention.csv
 * ==========================================================================*/
#include "common.cuh"
#include "attention.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <sys/stat.h>

/* Ejecuta el kernel n_iters veces y devuelve el tiempo medio por lanzamiento
 * en milisegundos. */
template <typename F>
static float time_kernel(F fn, int n_iters, int n_warmup)
{
    for (int i = 0; i < n_warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer t;
    t.start();
    for (int i = 0; i < n_iters; ++i) fn();
    const float ms = t.stop();
    return ms / (float)n_iters;
}

static float median(std::vector<float> v)
{
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

/* Comprueba que las dos variantes producen el mismo resultado. */
static float max_abs_diff(const float *d_a, const float *d_b, size_t n)
{
    std::vector<float> a(n), b(n);
    CUDA_CHECK(cudaMemcpy(a.data(), d_a, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), d_b, n * sizeof(float), cudaMemcpyDeviceToHost));
    float m = 0.0f;
    for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

int main(int argc, char **argv)
{
    int   batch    = 64;
    int   heads    = 4;
    int   d_model  = 64;
    int   n_iters  = 200;
    int   n_warmup = 20;
    int   n_repeats = 5;
    std::string out_dir = "../results";

    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "falta el valor de %s\n", k.c_str()); exit(1); }
            return argv[++i];
        };
        if      (k == "--batch")    batch    = atoi(next());
        else if (k == "--heads")    heads    = atoi(next());
        else if (k == "--d-model")  d_model  = atoi(next());
        else if (k == "--iters")    n_iters  = atoi(next());
        else if (k == "--warmup")   n_warmup = atoi(next());
        else if (k == "--repeats")  n_repeats = atoi(next());
        else if (k == "--out-dir")  out_dir  = next();
        else { fprintf(stderr, "opcion desconocida: %s\n", k.c_str()); exit(1); }
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | %d SMs | ancho de banda teorico %.0f GB/s\n",
           prop.name, prop.multiProcessorCount,
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("batch=%d heads=%d d_model=%d | iteraciones=%d calentamiento=%d repeticiones=%d\n\n",
           batch, heads, d_model, n_iters, n_warmup, n_repeats);

    const int Dh = d_model / heads;
    const int BH = batch * heads;

    /* Los tres tamanos de parche del experimento 1. */
    const int patches[3] = {4, 7, 14};

    mkdir(out_dir.c_str(), 0755);
    const std::string csv_path = out_dir + "/bench_attention.csv";
    FILE *fcsv = fopen(csv_path.c_str(), "w");
    if (!fcsv) { fprintf(stderr, "no se pudo escribir %s\n", csv_path.c_str()); return 1; }
    fprintf(fcsv, "kernel,patch,n_tokens,batch,heads,head_dim,"
                  "ms_global,ms_shared,speedup,gflops_global,gflops_shared,max_abs_diff\n");

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    for (int pi = 0; pi < 3; ++pi) {
        const int patch = patches[pi];
        const int n_patches = (28 / patch) * (28 / patch);
        const int T = n_patches + 1;

        const size_t n_qkv    = (size_t)BH * T * Dh;
        const size_t n_scores = (size_t)BH * T * T;

        float *q, *k, *v, *p_scores, *out_g, *out_s;
        CUDA_CHECK(cudaMalloc(&q, n_qkv * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&k, n_qkv * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&v, n_qkv * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&p_scores, n_scores * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&out_g, std::max(n_scores, n_qkv) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&out_s, std::max(n_scores, n_qkv) * sizeof(float)));

        /* Datos aleatorios realistas (activaciones normalizadas por LayerNorm). */
        {
            std::vector<float> h(std::max(n_qkv, n_scores));
            for (size_t i = 0; i < n_qkv; ++i) h[i] = nd(rng);
            CUDA_CHECK(cudaMemcpy(q, h.data(), n_qkv * sizeof(float), cudaMemcpyHostToDevice));
            for (size_t i = 0; i < n_qkv; ++i) h[i] = nd(rng);
            CUDA_CHECK(cudaMemcpy(k, h.data(), n_qkv * sizeof(float), cudaMemcpyHostToDevice));
            for (size_t i = 0; i < n_qkv; ++i) h[i] = nd(rng);
            CUDA_CHECK(cudaMemcpy(v, h.data(), n_qkv * sizeof(float), cudaMemcpyHostToDevice));
            /* p_scores hace de matriz de probabilidades: valores en [0,1]. */
            for (size_t i = 0; i < n_scores; ++i) h[i] = std::fabs(nd(rng)) / T;
            CUDA_CHECK(cudaMemcpy(p_scores, h.data(), n_scores * sizeof(float), cudaMemcpyHostToDevice));
        }

        const float scale = 1.0f / std::sqrt((float)Dh);

        /* ---------------- kernel 1: QK^T ---------------- */
        auto run_qk = [&](AttnMem mem, float *dst) {
            bmm_nt(q, k, dst, BH, T, T, Dh, T * Dh, T * Dh, T * T, scale, mem);
        };
        run_qk(ATTN_GLOBAL, out_g);
        run_qk(ATTN_SHARED, out_s);
        CUDA_CHECK(cudaDeviceSynchronize());
        const float diff_qk = max_abs_diff(out_g, out_s, n_scores);

        std::vector<float> tg, ts;
        for (int r = 0; r < n_repeats; ++r) {
            tg.push_back(time_kernel([&]{ run_qk(ATTN_GLOBAL, out_g); }, n_iters, n_warmup));
            ts.push_back(time_kernel([&]{ run_qk(ATTN_SHARED, out_s); }, n_iters, n_warmup));
        }
        const float qk_g = median(tg), qk_s = median(ts);
        /* 2*M*N*K operaciones de punto flotante por matriz, BH matrices. */
        const double flops_qk = 2.0 * BH * (double)T * T * Dh;

        printf("parche %2dx%-2d T=%2d | QK^T  global %8.4f ms  shared %8.4f ms  speedup %.2fx  (dif max %.2e)\n",
               patch, patch, T, qk_g, qk_s, qk_g / qk_s, diff_qk);
        fprintf(fcsv, "QK^T,%d,%d,%d,%d,%d,%.6f,%.6f,%.4f,%.3f,%.3f,%.3e\n",
                patch, T, batch, heads, Dh, qk_g, qk_s, qk_g / qk_s,
                flops_qk / (qk_g * 1e6), flops_qk / (qk_s * 1e6), diff_qk);

        /* ---------------- kernel 2: P*V ---------------- */
        auto run_pv = [&](AttnMem mem, float *dst) {
            bmm_nn(p_scores, v, dst, BH, T, Dh, T, T * T, T * Dh, T * Dh, 1.0f, mem);
        };
        run_pv(ATTN_GLOBAL, out_g);
        run_pv(ATTN_SHARED, out_s);
        CUDA_CHECK(cudaDeviceSynchronize());
        const float diff_pv = max_abs_diff(out_g, out_s, n_qkv);

        tg.clear(); ts.clear();
        for (int r = 0; r < n_repeats; ++r) {
            tg.push_back(time_kernel([&]{ run_pv(ATTN_GLOBAL, out_g); }, n_iters, n_warmup));
            ts.push_back(time_kernel([&]{ run_pv(ATTN_SHARED, out_s); }, n_iters, n_warmup));
        }
        const float pv_g = median(tg), pv_s = median(ts);
        const double flops_pv = 2.0 * BH * (double)T * Dh * T;

        printf("parche %2dx%-2d T=%2d | P*V   global %8.4f ms  shared %8.4f ms  speedup %.2fx  (dif max %.2e)\n\n",
               patch, patch, T, pv_g, pv_s, pv_g / pv_s, diff_pv);
        fprintf(fcsv, "PxV,%d,%d,%d,%d,%d,%.6f,%.6f,%.4f,%.3f,%.3f,%.3e\n",
                patch, T, batch, heads, Dh, pv_g, pv_s, pv_g / pv_s,
                flops_pv / (pv_g * 1e6), flops_pv / (pv_s * 1e6), diff_pv);

        cudaFree(q); cudaFree(k); cudaFree(v);
        cudaFree(p_scores); cudaFree(out_g); cudaFree(out_s);
    }

    fclose(fcsv);
    printf("CSV escrito en %s\n", csv_path.c_str());
    return 0;
}
