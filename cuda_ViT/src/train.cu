/* ============================================================================
 * train.cu -- Entrenamiento y evaluacion del ViT en MNIST.
 *
 * Uso tipico:
 *   ./vit_train --data-dir ../data --patch 7 --train-n 5000 --test-n 2000 \
 *               --epochs 12 --batch 64 --lr 3e-4 --attn-mem shared --tag p7_shared
 *
 * Produce dos CSV en results/:
 *   history_<tag>.csv  -> una fila por epoca (curvas de perdida y accuracy)
 *   summary_<tag>.csv  -> una fila con la configuracion y las metricas finales
 * ==========================================================================*/
#include "common.cuh"
#include "vit.cuh"
#include "ops.cuh"

extern "C" {
#include "data_loader.h"
}

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <sys/stat.h>

/* --------------------------------------------------------------------------
 * Construccion del batch en GPU.
 * Todo el subconjunto vive permanentemente en memoria de device; cada batch se
 * arma con un gather usando los indices barajados. Asi se evita una copia
 * host->device por iteracion, que con batches tan pequenos dominaria el tiempo.
 * ------------------------------------------------------------------------*/
__global__ void gather_batch_kernel(const float *__restrict__ all_img,
                                    const int *__restrict__ all_lab,
                                    const int *__restrict__ idx,
                                    float *__restrict__ out_img,
                                    int *__restrict__ out_lab,
                                    int n_valid, int img_pixels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_valid * img_pixels) return;
    int p = i % img_pixels;
    int b = i / img_pixels;
    out_img[i] = all_img[(size_t)idx[b] * img_pixels + p];
    if (p == 0) out_lab[b] = all_lab[idx[b]];
}

struct Args {
    std::string data_dir = "../data";
    std::string out_dir  = "../results";
    std::string tag      = "";
    int   patch    = 7;
    int   d_model  = 64;
    int   n_heads  = 4;
    int   n_blocks = 2;
    int   mlp_hidden = 128;
    int   train_n  = 5000;
    int   test_n   = 2000;
    int   epochs   = 15;
    int   batch    = 64;
    /* lr = 1e-3 verificado con la referencia en PyTorch: con 3e-4 las tres
     * configuraciones quedan subentrenadas a 15 epocas y la comparacion entre
     * tamanos de parche mediria velocidad de convergencia en vez de calidad. */
    float lr       = 1e-3f;
    float wd       = 0.01f;
    unsigned long long seed = 1234;
    AttnMem mem    = ATTN_SHARED;
    /* Presupuesto de tiempo: si la primera epoca tarda mas que esto, el
     * programa reduce el subconjunto y reinicia (ver --auto-shrink). */
    float max_epoch_sec = 30.0f;
    int   auto_shrink   = 1;
};

static void parse_args(int argc, char **argv, Args &a)
{
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "falta el valor de %s\n", k.c_str()); exit(1); }
            return argv[++i];
        };
        if      (k == "--data-dir")  a.data_dir = next();
        else if (k == "--out-dir")   a.out_dir  = next();
        else if (k == "--tag")       a.tag      = next();
        else if (k == "--patch")     a.patch    = atoi(next());
        else if (k == "--d-model")   a.d_model  = atoi(next());
        else if (k == "--heads")     a.n_heads  = atoi(next());
        else if (k == "--blocks")    a.n_blocks = atoi(next());
        else if (k == "--mlp-hidden")a.mlp_hidden = atoi(next());
        else if (k == "--train-n")   a.train_n  = atoi(next());
        else if (k == "--test-n")    a.test_n   = atoi(next());
        else if (k == "--epochs")    a.epochs   = atoi(next());
        else if (k == "--batch")     a.batch    = atoi(next());
        else if (k == "--lr")        a.lr       = (float)atof(next());
        else if (k == "--wd")        a.wd       = (float)atof(next());
        else if (k == "--seed")      a.seed     = strtoull(next(), nullptr, 10);
        else if (k == "--max-epoch-sec") a.max_epoch_sec = (float)atof(next());
        else if (k == "--no-auto-shrink") a.auto_shrink = 0;
        else if (k == "--attn-mem") {
            std::string v = next();
            if (v == "shared") a.mem = ATTN_SHARED;
            else if (v == "global") a.mem = ATTN_GLOBAL;
            else { fprintf(stderr, "--attn-mem debe ser 'global' o 'shared'\n"); exit(1); }
        }
        else if (k == "--help") {
            printf("opciones: --data-dir --out-dir --tag --patch --d-model --heads --blocks\n"
                   "          --mlp-hidden --train-n --test-n --epochs --batch --lr --wd --seed\n"
                   "          --attn-mem {global|shared} --max-epoch-sec --no-auto-shrink\n");
            exit(0);
        }
        else { fprintf(stderr, "opcion desconocida: %s\n", k.c_str()); exit(1); }
    }
    if (28 % a.patch != 0) {
        fprintf(stderr, "el tamano de parche debe dividir a 28 (4, 7 o 14)\n");
        exit(1);
    }
    if (a.d_model % a.n_heads != 0) {
        fprintf(stderr, "d_model debe ser divisible por el numero de cabezas\n");
        exit(1);
    }
    if (a.tag.empty())
        a.tag = "p" + std::to_string(a.patch) + "_" +
                (a.mem == ATTN_SHARED ? "shared" : "global");
}

/* Datos de un split residentes en GPU. */
struct DeviceSplit {
    float *images = nullptr;
    int   *labels = nullptr;
    int    n = 0;
};

static void upload_split(const MnistData &h, DeviceSplit &d)
{
    const size_t npix = (size_t)h.rows * h.cols;
    d.n = h.n_samples;
    CUDA_CHECK(cudaMalloc(&d.images, (size_t)d.n * npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.labels, (size_t)d.n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d.images, h.images, (size_t)d.n * npix * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.labels, h.labels, (size_t)d.n * sizeof(int), cudaMemcpyHostToDevice));
}

struct EpochRow {
    int   epoch;
    float train_loss, train_acc, test_loss, test_acc, epoch_sec;
};

/* Evaluacion sobre un split completo: devuelve perdida media, accuracy y el
 * tiempo medio de inferencia por batch en milisegundos. */
static void evaluate(ViT &m, DeviceSplit &split, int *d_idx, const std::vector<int> &order,
                     int batch, int img_pixels, float &out_loss, float &out_acc,
                     float &out_ms_per_batch)
{
    const int n = split.n;
    const int n_batches = (n + batch - 1) / batch;
    double loss_sum = 0.0;
    int total = 0;

    CUDA_CHECK(cudaMemset(m.d_correct, 0, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_idx, order.data(), (size_t)n * sizeof(int), cudaMemcpyHostToDevice));

    CudaTimer timer;
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.start();

    for (int b = 0; b < n_batches; ++b) {
        int nv = std::min(batch, n - b * batch);
        const int thr = 256, work = nv * img_pixels;
        gather_batch_kernel<<<ceil_div(work, thr), thr>>>(
            split.images, split.labels, d_idx + (size_t)b * batch,
            m.d_images, m.d_labels, nv, img_pixels);
        CUDA_CHECK_KERNEL();

        float l = vit_forward(m, m.d_images, m.d_labels, nv, true);
        loss_sum += (double)l * nv;
        total += nv;
        count_correct(m.probs_out, m.d_labels, m.d_correct, nv, m.cfg.n_classes);
    }

    float ms = timer.stop();

    int correct = 0;
    CUDA_CHECK(cudaMemcpy(&correct, m.d_correct, sizeof(int), cudaMemcpyDeviceToHost));
    out_loss = (float)(loss_sum / total);
    out_acc  = (float)correct / (float)total;
    out_ms_per_batch = ms / (float)n_batches;
}

int main(int argc, char **argv)
{
    Args args;
    parse_args(argc, argv, args);

    /* --- informacion de la GPU --- */
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | SMs: %d | shared/bloque: %zu KB | capacidad %d.%d\n",
           prop.name, prop.multiProcessorCount,
           prop.sharedMemPerBlock / 1024, prop.major, prop.minor);

    /* --- carga de datos --- */
    std::string p_tri = args.data_dir + "/train-images-idx3-ubyte";
    std::string p_trl = args.data_dir + "/train-labels-idx1-ubyte";
    std::string p_tei = args.data_dir + "/t10k-images-idx3-ubyte";
    std::string p_tel = args.data_dir + "/t10k-labels-idx1-ubyte";

    MnistData h_train{}, h_test{};
    if (mnist_load(p_tri.c_str(), p_trl.c_str(), args.train_n, &h_train)) return 1;
    if (mnist_load(p_tei.c_str(), p_tel.c_str(), args.test_n,  &h_test))  return 1;

    printf("datos: %d train / %d test, imagenes de %dx%d\n",
           h_train.n_samples, h_test.n_samples, h_train.rows, h_train.cols);
    /* Sanity check obligatorio antes de entrenar: si esto no se ve como un
     * digito, el parser esta mal y no tiene sentido seguir. */
    mnist_print_ascii(&h_train, 0);

    const int img_pixels = h_train.rows * h_train.cols;

    DeviceSplit d_train{}, d_test{};
    upload_split(h_train, d_train);
    upload_split(h_test,  d_test);

    int *d_idx_train = nullptr, *d_idx_test = nullptr;
    CUDA_CHECK(cudaMalloc(&d_idx_train, (size_t)d_train.n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_idx_test,  (size_t)d_test.n  * sizeof(int)));

    std::vector<int> order_train(d_train.n), order_test(d_test.n);
    std::iota(order_train.begin(), order_train.end(), 0);
    std::iota(order_test.begin(),  order_test.end(),  0);

    const double mem_before = gpu_mem_used_mib();

    /* ======================= bucle de entrenamiento ======================= */
    int attempt = 0;
    int train_n_eff = d_train.n;
    std::vector<EpochRow> history;
    float best_test_acc = 0.0f, final_test_acc = 0.0f, final_test_loss = 0.0f;
    float infer_ms_per_batch = 0.0f;
    double total_train_sec = 0.0, mem_used = 0.0;
    int n_params = 0;

    while (true) {
        history.clear();
        total_train_sec = 0.0;
        best_test_acc = 0.0f;

        ViTConfig cfg;
        cfg.img_size = h_train.rows;
        cfg.patch    = args.patch;
        cfg.d_model  = args.d_model;
        cfg.n_heads  = args.n_heads;
        cfg.n_blocks = args.n_blocks;
        cfg.mlp_hidden = args.mlp_hidden;
        cfg.batch    = args.batch;
        cfg.attn_mem = args.mem;

        ViT model;
        vit_init(model, cfg, args.seed);
        n_params = model.n_params;
        mem_used = gpu_mem_used_mib() - mem_before;

        printf("\nconfiguracion: parche %dx%d -> %d parches (%d tokens con CLS)\n",
               cfg.patch, cfg.patch, cfg.n_patches(), cfg.n_tokens());
        printf("d_model=%d heads=%d bloques=%d mlp=%d | parametros=%d | memoria GPU=%.1f MiB\n",
               cfg.d_model, cfg.n_heads, cfg.n_blocks, cfg.mlp_hidden, n_params, mem_used);
        printf("atencion: memoria %s | train_n=%d test_n=%d batch=%d epocas=%d lr=%g\n",
               (cfg.attn_mem == ATTN_SHARED ? "COMPARTIDA (tiling)" : "GLOBAL (sin tiling)"),
               train_n_eff, d_test.n, args.batch, args.epochs, args.lr);

        std::mt19937 shuffler((unsigned int)args.seed);
        std::vector<int> epoch_order(order_train.begin(), order_train.begin() + train_n_eff);

        bool restart = false;

        for (int ep = 1; ep <= args.epochs; ++ep) {
            std::shuffle(epoch_order.begin(), epoch_order.end(), shuffler);
            CUDA_CHECK(cudaMemcpy(d_idx_train, epoch_order.data(),
                                  (size_t)train_n_eff * sizeof(int), cudaMemcpyHostToDevice));

            const int n_batches = (train_n_eff + args.batch - 1) / args.batch;
            double loss_sum = 0.0;
            int total = 0;
            CUDA_CHECK(cudaMemset(model.d_correct, 0, sizeof(int)));

            CudaTimer timer;
            CUDA_CHECK(cudaDeviceSynchronize());
            timer.start();

            for (int b = 0; b < n_batches; ++b) {
                int nv = std::min(args.batch, train_n_eff - b * args.batch);

                const int thr = 256, work = nv * img_pixels;
                gather_batch_kernel<<<ceil_div(work, thr), thr>>>(
                    d_train.images, d_train.labels, d_idx_train + (size_t)b * args.batch,
                    model.d_images, model.d_labels, nv, img_pixels);
                CUDA_CHECK_KERNEL();

                float l = vit_forward(model, model.d_images, model.d_labels, nv, true);
                count_correct(model.probs_out, model.d_labels, model.d_correct, nv, cfg.n_classes);

                vit_zero_grad(model);
                vit_backward(model, nv);
                vit_adam(model, args.lr, args.wd);

                loss_sum += (double)l * nv;
                total += nv;
            }

            const float epoch_ms = timer.stop();
            const float epoch_sec = epoch_ms / 1000.0f;
            total_train_sec += epoch_sec;

            int correct = 0;
            CUDA_CHECK(cudaMemcpy(&correct, model.d_correct, sizeof(int), cudaMemcpyDeviceToHost));
            const float train_loss = (float)(loss_sum / total);
            const float train_acc  = (float)correct / (float)total;

            float te_loss, te_acc, te_ms;
            evaluate(model, d_test, d_idx_test, order_test, args.batch, img_pixels,
                     te_loss, te_acc, te_ms);
            infer_ms_per_batch = te_ms;
            best_test_acc = std::max(best_test_acc, te_acc);
            final_test_acc = te_acc; final_test_loss = te_loss;

            printf("epoca %2d/%d | perdida %.4f acc %.4f | test perdida %.4f acc %.4f | %.2f s\n",
                   ep, args.epochs, train_loss, train_acc, te_loss, te_acc, epoch_sec);
            fflush(stdout);

            history.push_back({ep, train_loss, train_acc, te_loss, te_acc, epoch_sec});

            /* ---- guardia de presupuesto de tiempo ----
             * Se comprueba TRAS LA PRIMERA EPOCA, antes de gastar el resto.
             * Si se excede, se reduce el subconjunto a la mitad y se reinicia
             * desde cero (pesos incluidos) para que la corrida siga siendo
             * coherente, informando exactamente que cambio y por que. */
            if (ep == 1 && epoch_sec > args.max_epoch_sec) {
                if (args.auto_shrink && attempt < 2 && train_n_eff > 1000) {
                    const int nuevo = std::max(1000, train_n_eff / 2);
                    printf("\n*** AVISO DE PRESUPUESTO DE TIEMPO ***\n"
                           "la epoca 1 tardo %.1f s, por encima del limite de %.1f s.\n"
                           "cambio aplicado: train_n de %d a %d (mitad), reiniciando desde cero.\n"
                           "motivo: %d epocas a este ritmo darian ~%.1f min, fuera del presupuesto\n"
                           "de 1-5 min por corrida.\n\n",
                           epoch_sec, args.max_epoch_sec, train_n_eff, nuevo,
                           args.epochs, epoch_sec * args.epochs / 60.0f);
                    train_n_eff = nuevo;
                    attempt++;
                    restart = true;
                    break;
                } else {
                    printf("\n*** AVISO: la epoca 1 tardo %.1f s (limite %.1f s) y no se puede\n"
                           "reducir mas automaticamente. Se continua, pero considera bajar\n"
                           "--d-model, --blocks o --train-n a mano.\n\n",
                           epoch_sec, args.max_epoch_sec);
                }
            }
        }

        vit_free(model);
        if (!restart) break;
    }

    /* ============================ salida CSV ============================= */
    mkdir(args.out_dir.c_str(), 0755);   /* si ya existe, falla en silencio */

    std::string hist_path = args.out_dir + "/history_" + args.tag + ".csv";
    FILE *fh = fopen(hist_path.c_str(), "w");
    if (!fh) { fprintf(stderr, "no se pudo escribir %s\n", hist_path.c_str()); return 1; }
    fprintf(fh, "epoch,train_loss,train_acc,test_loss,test_acc,epoch_sec\n");
    for (const auto &r : history)
        fprintf(fh, "%d,%.6f,%.6f,%.6f,%.6f,%.4f\n",
                r.epoch, r.train_loss, r.train_acc, r.test_loss, r.test_acc, r.epoch_sec);
    fclose(fh);

    const int n_patches = (28 / args.patch) * (28 / args.patch);
    std::string sum_path = args.out_dir + "/summary_" + args.tag + ".csv";
    FILE *fs = fopen(sum_path.c_str(), "w");
    if (!fs) { fprintf(stderr, "no se pudo escribir %s\n", sum_path.c_str()); return 1; }
    fprintf(fs, "tag,patch,n_patches,n_tokens,attn_mem,d_model,heads,blocks,params,"
                "train_n,test_n,batch,epochs,test_acc,best_test_acc,test_loss,"
                "train_total_sec,epoch_sec_mean,infer_ms_per_batch,gpu_mem_mib\n");
    double mean_ep = 0.0;
    for (const auto &r : history) mean_ep += r.epoch_sec;
    if (!history.empty()) mean_ep /= history.size();
    fprintf(fs, "%s,%d,%d,%d,%s,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.3f,%.3f,%.4f,%.1f\n",
            args.tag.c_str(), args.patch, n_patches, n_patches + 1,
            (args.mem == ATTN_SHARED ? "shared" : "global"),
            args.d_model, args.n_heads, args.n_blocks, n_params,
            train_n_eff, d_test.n, args.batch, args.epochs,
            final_test_acc, best_test_acc, final_test_loss,
            total_train_sec, mean_ep, infer_ms_per_batch, mem_used);
    fclose(fs);

    printf("\nresultados finales: accuracy en test %.4f (mejor %.4f)\n", final_test_acc, best_test_acc);
    printf("tiempo total de entrenamiento %.1f s | media por epoca %.2f s | inferencia %.3f ms/batch\n",
           total_train_sec, mean_ep, infer_ms_per_batch);
    printf("CSV escritos en %s y %s\n", hist_path.c_str(), sum_path.c_str());

    cudaFree(d_train.images); cudaFree(d_train.labels);
    cudaFree(d_test.images);  cudaFree(d_test.labels);
    cudaFree(d_idx_train);    cudaFree(d_idx_test);
    mnist_free(&h_train);
    mnist_free(&h_test);
    return 0;
}
