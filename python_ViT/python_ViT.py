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
