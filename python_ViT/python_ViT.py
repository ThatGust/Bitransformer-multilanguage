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


