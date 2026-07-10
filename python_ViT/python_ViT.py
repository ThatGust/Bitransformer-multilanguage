import numpy as np
import gzip
import time
import csv
import os



SEED = 42
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

RUTA_IMAGENES = "gzip/emnist-balanced-train-images-idx3-ubyte.gz"
RUTA_LABELS   = "gzip/emnist-balanced-train-labels-idx1-ubyte.gz"

RUTA_CSV_METRICAS = "./metricas_entrenamiento.csv"
RUTA_PESOS_FINALES = "./modelo_final.npz"

EPOCAS = 60
MUESTRAS_TOTAL = 10000      
FRACCION_VAL = 0.10          
TASA_APRENDIZAJE = 0.0005

SOM_LR_INICIAL = 0.5
SOM_LR_FINAL = 0.01
SIGMA_INICIAL = 3.0          
SIGMA_FINAL = 0.5
EPOCAS_ENTRENAMIENTO_SOM = 25  

NUM_CLASES = 47




def cargar_imagenes(ruta):
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró el archivo de imágenes en '{ruta}'. "
            "Ajusta la constante RUTA_IMAGENES al path correcto en tu sistema."
        )
    with gzip.open(ruta, 'rb') as f:
        f.read(16)
        imagenes = np.frombuffer(f.read(), dtype=np.uint8)
        return imagenes.reshape(-1, 28, 28)


def cargar_labels(ruta):
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró el archivo de labels en '{ruta}'. "
            "Ajusta la constante RUTA_LABELS al path correcto en tu sistema."
        )
    with gzip.open(ruta, 'rb') as f:
        f.read(8)
        return np.frombuffer(f.read(), dtype=np.uint8)

def extract_patches(image, patch_size=7):
    H, W = image.shape
    shape = (H // patch_size, W // patch_size, patch_size, patch_size)
    strides = (image.strides[0] * patch_size, image.strides[1] * patch_size, image.strides[0], image.strides[1])
    return np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)


def flatten_patches(patches):
    num_patches_h, num_patches_w, p_h, p_w = patches.shape
    return patches.reshape(num_patches_h * num_patches_w, p_h * p_w)

class Linear:
    def __init__(self, d_in, d_out):
        self.W = np.random.randn(d_in, d_out) * np.sqrt(2.0 / d_in)
        self.b = np.zeros(d_out)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout, lr):
        dW = self.x.T @ dout
        db = np.sum(dout, axis=0)
        dx = dout @ self.W.T
        self.W -= lr * dW
        self.b -= lr * db
        return dx


class LayerNorm:
    def __init__(self, d, eps=1e-5):
        self.gamma = np.ones(d)
        self.beta = np.zeros(d)
        self.eps = eps

    def forward(self, x):
        self.x = x
        self.mu = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.std = np.sqrt(self.var + self.eps)
        self.x_norm = (x - self.mu) / self.std
        return self.gamma * self.x_norm + self.beta

    def backward(self, dout, lr):
        D = self.x.shape[-1]
        dx_norm = dout * self.gamma
        dvar = np.sum(dx_norm * (self.x - self.mu) * -0.5 * (self.var + self.eps) ** (-1.5), axis=-1, keepdims=True)
        dmu = np.sum(dx_norm * -1.0 / self.std, axis=-1, keepdims=True) + dvar * np.mean(-2.0 * (self.x - self.mu), axis=-1, keepdims=True)
        dx = (dx_norm / self.std) + (dvar * 2.0 * (self.x - self.mu) / D) + (dmu / D)

        self.gamma -= lr * np.sum(dout * self.x_norm, axis=0)
        self.beta -= lr * np.sum(dout, axis=0)
        return dx


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout, lr):
        return dout * self.mask




class KohonenPatchEmbedding:


    def __init__(self, patch_dim=49, grid_size=8, hidden_dim=64, num_patches=16):
        self.grid_size = grid_size
        num_som_nodes = grid_size * grid_size

        self.som_weights = np.random.rand(num_som_nodes, patch_dim)

        gy, gx = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing='ij')
        self.grid_coords = np.stack([gy.ravel(), gx.ravel()], axis=1).astype(np.float32)  

        self.proj = Linear(num_som_nodes, hidden_dim)
        self.cls_token = np.random.randn(1, hidden_dim) * 0.01
        self.pos_embed = np.random.randn(num_patches + 1, hidden_dim) * 0.01

    def forward(self, patches, train_som=True, som_lr=0.1, sigma=1.5):
        if train_som:
            for patch in patches:
                dist = np.linalg.norm(self.som_weights - patch, axis=1)
                bmu_idx = np.argmin(dist)

                
                grid_dist_sq = np.sum((self.grid_coords - self.grid_coords[bmu_idx]) ** 2, axis=1)
                vecindad = np.exp(-grid_dist_sq / (2.0 * sigma ** 2 + 1e-8))

                
                self.som_weights += (som_lr * vecindad)[:, None] * (patch - self.som_weights)


        dists = np.linalg.norm(self.som_weights[None, :, :] - patches[:, None, :], axis=2)  
        dists_norm = dists / (np.max(dists, axis=1, keepdims=True) + 1e-8)
        som_activations = np.exp(-dists_norm ** 2)

        patch_embs = self.proj.forward(som_activations)
        embs = np.vstack([self.cls_token, patch_embs])
        return embs + self.pos_embed

    def backward(self, dout, lr):
        self.pos_embed -= lr * dout
        self.cls_token -= lr * dout[0:1, :]
        
        _ = self.proj.backward(dout[1:, :], lr)


class ModernHopfieldLayer:
    def __init__(self, d_model):
        scale = np.sqrt(1.0 / d_model)
        self.W_state = np.random.randn(d_model, d_model) * scale
        self.W_pattern_in = np.random.randn(d_model, d_model) * scale
        self.W_pattern_out = np.random.randn(d_model, d_model) * scale
        self.W_proj = np.random.randn(d_model, d_model) * scale

    def forward(self, x):
        self.x = x
        self.State = x @ self.W_state
        self.Pattern_In = x @ self.W_pattern_in
        self.Pattern_Out = x @ self.W_pattern_out

        beta = 1.0 / np.sqrt(self.State.shape[-1])
        self.energy_scores = (self.State @ self.Pattern_In.T) * beta

        exps = np.exp(self.energy_scores - np.max(self.energy_scores, axis=-1, keepdims=True))
        self.Hopfield_Matrix = exps / np.sum(exps, axis=-1, keepdims=True)

        self.new_state = self.Hopfield_Matrix @ self.Pattern_Out
        return self.new_state @ self.W_proj

    def backward(self, dout, lr):
        dW_proj = self.new_state.T @ dout
        dContext = dout @ self.W_proj.T

        dPatOut = self.Hopfield_Matrix.T @ dContext
        dW_pat_out = self.x.T @ dPatOut

        dHop = dContext @ self.Pattern_Out.T
        beta = 1.0 / np.sqrt(self.State.shape[-1])
        dScores = self.Hopfield_Matrix * (dHop - np.sum(dHop * self.Hopfield_Matrix, axis=-1, keepdims=True)) * beta

        dState = dScores @ self.Pattern_In
        dW_state = self.x.T @ dState

        dPatIn = dScores.T @ self.State
        dW_pat_in = self.x.T @ dPatIn

        dx = dState @ self.W_state.T + dPatIn @ self.W_pattern_in.T + dPatOut @ self.W_pattern_out.T

        self.W_proj -= lr * dW_proj
        self.W_pattern_out -= lr * dW_pat_out
        self.W_state -= lr * dW_state
        self.W_pattern_in -= lr * dW_pat_in
        return dx



class HopfieldKohonenViT:
    def __init__(self):
        self.embed = KohonenPatchEmbedding(patch_dim=49, grid_size=8, hidden_dim=64, num_patches=16)

        self.ln1 = LayerNorm(64)
        self.hopfield = ModernHopfieldLayer(64)
        self.ln2 = LayerNorm(64)
        self.mlp_l1 = Linear(64, 128)
        self.mlp_relu = ReLU()
        self.mlp_l2 = Linear(128, 64)

        self.norm_final = LayerNorm(64)
        self.head = Linear(64, NUM_CLASES)

    def forward(self, image, train=True, som_lr=0.1, sigma=1.5):
        patches = flatten_patches(extract_patches(image, 7))

        x = self.embed.forward(patches, train_som=train, som_lr=som_lr, sigma=sigma)

        self.res1 = x
        x_norm1 = self.ln1.forward(x)
        hopfield_out = self.hopfield.forward(x_norm1)
        self.res2 = self.res1 + hopfield_out  

     
        res2_cls = self.res2[0:1, :]
        x_norm2_cls = self.ln2.forward(res2_cls)
        mlp_out_cls = self.mlp_l2.forward(self.mlp_relu.forward(self.mlp_l1.forward(x_norm2_cls)))
        self.final_cls = res2_cls + mlp_out_cls

        return self.head.forward(self.norm_final.forward(self.final_cls))[0]

    def backward(self, d_logits, lr):
        d_cls_norm = self.head.backward(d_logits.reshape(1, -1), lr)
        d_final_cls = self.norm_final.backward(d_cls_norm, lr)

        d_ln2_in = self.mlp_l1.backward(self.mlp_relu.backward(self.mlp_l2.backward(d_final_cls, lr), lr), lr)
        d_res2_cls = d_final_cls + self.ln2.backward(d_ln2_in, lr)

        d_hop_out = np.zeros_like(self.res2)
        d_hop_out[0:1, :] = d_res2_cls

        d_ln1_in = self.hopfield.backward(d_hop_out, lr)
        d_embed_out = d_hop_out + self.ln1.backward(d_ln1_in, lr)

        self.embed.backward(d_embed_out, lr)

    def guardar_pesos(self, ruta):
        np.savez(
            ruta,
            som_weights=self.embed.som_weights,
            proj_W=self.embed.proj.W, proj_b=self.embed.proj.b,
            cls_token=self.embed.cls_token, pos_embed=self.embed.pos_embed,
            ln1_gamma=self.ln1.gamma, ln1_beta=self.ln1.beta,
            hop_W_state=self.hopfield.W_state, hop_W_pattern_in=self.hopfield.W_pattern_in,
            hop_W_pattern_out=self.hopfield.W_pattern_out, hop_W_proj=self.hopfield.W_proj,
            ln2_gamma=self.ln2.gamma, ln2_beta=self.ln2.beta,
            mlp_l1_W=self.mlp_l1.W, mlp_l1_b=self.mlp_l1.b,
            mlp_l2_W=self.mlp_l2.W, mlp_l2_b=self.mlp_l2.b,
            norm_final_gamma=self.norm_final.gamma, norm_final_beta=self.norm_final.beta,
            head_W=self.head.W, head_b=self.head.b,
        )

def softmax_cross_entropy(logits, label):
    exps = np.exp(logits - np.max(logits))
    probs = exps / np.sum(exps)
    loss = -np.log(probs[label] + 1e-12)
    d_logits = probs.copy()
    d_logits[label] -= 1.0
    return loss, probs, d_logits



def precision_macro(y_true, y_pred, num_clases):
 
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    precisiones = np.zeros(num_clases)
    for c in range(num_clases):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        if tp + fp > 0:
            precisiones[c] = tp / (tp + fp)
    return float(np.mean(precisiones))

def main():
    print("Cargando EMNIST...")
    X_full = cargar_imagenes(RUTA_IMAGENES).astype(np.float32) / 255.0
    y_full = cargar_labels(RUTA_LABELS)

    n_disponibles = min(MUESTRAS_TOTAL, len(X_full))
    indices = rng.permutation(len(X_full))[:n_disponibles]

    n_val = int(n_disponibles * FRACCION_VAL)
    idx_val = indices[:n_val]
    idx_train = indices[n_val:]

    X_train, y_train = X_full[idx_train], y_full[idx_train]
    X_val, y_val = X_full[idx_val], y_full[idx_val]

    print(f"Muestras de entrenamiento: {len(X_train)} | Muestras de validación: {len(X_val)}")

    modelo = HopfieldKohonenViT()

    
    with open(RUTA_CSV_METRICAS, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "epoca", "tiempo_seg",
            "train_loss", "train_acc", "train_precision",
            "val_loss", "val_acc", "val_precision",
            "som_lr", "sigma", "som_entrenandose"
        ])

    print("\n--- INICIANDO ENTRENAMIENTO HÍBRIDO (KOHONEN + HOPFIELD) ---")

    for epoca in range(EPOCAS):
        inicio = time.time()

        
        progreso = epoca / max(EPOCAS_ENTRENAMIENTO_SOM - 1, 1)
        progreso = min(progreso, 1.0)
        som_lr = SOM_LR_INICIAL * (SOM_LR_FINAL / SOM_LR_INICIAL) ** progreso
        sigma = SIGMA_INICIAL * (SIGMA_FINAL / SIGMA_INICIAL) ** progreso
        som_entrenandose = epoca < EPOCAS_ENTRENAMIENTO_SOM

        
        orden = rng.permutation(len(X_train))

        perdida_total = 0.0
        etiquetas_reales_train = np.empty(len(X_train), dtype=np.int64)
        etiquetas_pred_train = np.empty(len(X_train), dtype=np.int64)

        for pos, i in enumerate(orden):
            imagen = X_train[i]
            label = int(y_train[i])

            logits = modelo.forward(imagen, train=som_entrenandose, som_lr=som_lr, sigma=sigma)
            loss, probs, d_logits = softmax_cross_entropy(logits, label)
            perdida_total += loss

            pred = int(np.argmax(probs))
            etiquetas_reales_train[pos] = label
            etiquetas_pred_train[pos] = pred

            modelo.backward(d_logits, lr=TASA_APRENDIZAJE)

            if (pos + 1) % 5000 == 0:
                print(f"Época {epoca + 1} | Paso {pos + 1}/{len(X_train)} | Loss actual: {loss:.4f}")

        train_loss = perdida_total / len(X_train)
        train_acc = float(np.mean(etiquetas_pred_train == etiquetas_reales_train))
        train_prec = precision_macro(etiquetas_reales_train, etiquetas_pred_train, NUM_CLASES)
        
