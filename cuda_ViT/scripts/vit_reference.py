#!/usr/bin/env python3
"""
vit_reference.py -- ViT minimo en PyTorch, replica EXACTA de la arquitectura
implementada en CUDA (mismo numero de bloques, misma dimension de embedding,
mismo pre-norm, mismo optimizador, mismos hiperparametros).

NO forma parte de la entrega: es el control de sanidad. Sirve para responder
"que accuracy deberia dar esta arquitectura con este presupuesto de datos y
epocas?" antes de invertir tiempo depurando la version en CUDA. Si la version
CUDA queda muy por debajo de estos numeros, el problema esta en los kernels o
en el backward, no en la configuracion del modelo.

Lee los .ubyte directamente, sin torchvision. Funciona en CPU, en CUDA y en
MPS (Apple Silicon).

    python3 scripts/vit_reference.py --patch 7 --epochs 12
    python3 scripts/vit_reference.py --all      # los tres tamanos de parche
"""
import argparse
import json
import struct
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------- datos
def load_idx_images(path):
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 0x803, f"magic inesperado {magic:#x} en {path}"
        buf = f.read(n * rows * cols)
    x = torch.frombuffer(bytearray(buf), dtype=torch.uint8).float().div_(255.0)
    return x.view(n, rows, cols)


def load_idx_labels(path):
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 0x801, f"magic inesperado {magic:#x} en {path}"
        buf = f.read(n)
    return torch.frombuffer(bytearray(buf), dtype=torch.uint8).long()


# --------------------------------------------------------------------- modelo
class Block(nn.Module):
    """Bloque Transformer pre-norm, identico al implementado en CUDA."""

    def __init__(self, d, heads, hidden):
        super().__init__()
        self.ln1 = nn.LayerNorm(d, eps=1e-5)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d, eps=1e-5)
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)
        self.heads = heads
        self.dh = d // heads

    def forward(self, x):
        B, T, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, T, 3, self.heads, self.dh)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))   # [B,H,T,Dh]
        att = (q @ k.transpose(-2, -1)) * (self.dh ** -0.5)
        att = att.softmax(dim=-1)
        out = (att @ v).permute(0, 2, 1, 3).reshape(B, T, D)
        x = x + self.proj(out)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))
        return x


class ViT(nn.Module):
    def __init__(self, img=28, patch=7, d=64, heads=4, blocks=2, hidden=128, n_cls=10):
        super().__init__()
        self.patch = patch
        self.grid = img // patch
        self.n_patches = self.grid ** 2
        self.embed = nn.Linear(patch * patch, d)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches + 1, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, heads, hidden) for _ in range(blocks)])
        self.lnf = nn.LayerNorm(d, eps=1e-5)
        self.head = nn.Linear(d, n_cls)

    def forward(self, img):
        B = img.shape[0]
        p = self.patch
        # im2patch: mismo orden de parches que el kernel im2patch de CUDA
        x = img.unfold(1, p, p).unfold(2, p, p)           # [B,g,g,p,p]
        x = x.reshape(B, self.n_patches, p * p)
        x = self.embed(x)
        x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.lnf(x[:, 0]))


# ------------------------------------------------------------------ ejecucion
def run(args, patch, device, train, test):
    torch.manual_seed(args.seed)
    xtr, ytr = train
    xte, yte = test

    model = ViT(patch=patch, d=args.d_model, heads=args.heads,
                blocks=args.blocks, hidden=args.mlp_hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    n = xtr.shape[0]
    hist = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        tot_loss, correct = 0.0, 0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            xb, yb = xtr[idx], ytr[idx]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()

        model.eval()
        with torch.no_grad():
            te_logits = torch.cat([model(xte[i:i + 256]) for i in range(0, len(xte), 256)])
            te_loss = F.cross_entropy(te_logits, yte).item()
            te_acc = (te_logits.argmax(1) == yte).float().mean().item()

        hist.append(dict(epoch=ep, train_loss=tot_loss / n, train_acc=correct / n,
                         test_loss=te_loss, test_acc=te_acc))
        print(f"  epoca {ep:2d}/{args.epochs} | perdida {tot_loss/n:.4f} "
              f"acc {correct/n:.4f} | test perdida {te_loss:.4f} acc {te_acc:.4f}")

    dur = time.time() - t0
    return dict(patch=patch, n_patches=(28 // patch) ** 2, n_tokens=(28 // patch) ** 2 + 1,
                params=n_params, test_acc=hist[-1]["test_acc"],
                best_test_acc=max(h["test_acc"] for h in hist),
                seconds=dur, history=hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "pytorch_reference.json"))
    ap.add_argument("--patch", type=int, default=7)
    ap.add_argument("--all", action="store_true", help="corre los tres tamanos de parche")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--mlp-hidden", type=int, default=128)
    ap.add_argument("--train-n", type=int, default=5000)
    ap.add_argument("--test-n", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"dispositivo: {device}")

    d = Path(args.data_dir)
    xtr = load_idx_images(d / "train-images-idx3-ubyte")[:args.train_n].to(device)
    ytr = load_idx_labels(d / "train-labels-idx1-ubyte")[:args.train_n].to(device)
    xte = load_idx_images(d / "t10k-images-idx3-ubyte")[:args.test_n].to(device)
    yte = load_idx_labels(d / "t10k-labels-idx1-ubyte")[:args.test_n].to(device)
    print(f"datos: {len(xtr)} train / {len(xte)} test")

    patches = [4, 7, 14] if args.all else [args.patch]
    results = []
    for p in patches:
        print(f"\n--- parche {p}x{p} -> {(28//p)**2} parches, {(28//p)**2 + 1} tokens ---")
        r = run(args, p, device, (xtr, ytr), (xte, yte))
        print(f"  => accuracy final {r['test_acc']:.4f} | mejor {r['best_test_acc']:.4f} "
              f"| {r['params']} parametros | {r['seconds']:.1f} s")
        results.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(device=str(device), train_n=args.train_n, test_n=args.test_n,
                epochs=args.epochs, batch=args.batch, lr=args.lr, wd=args.wd,
                d_model=args.d_model, heads=args.heads, blocks=args.blocks,
                mlp_hidden=args.mlp_hidden, seed=args.seed, runs=results)
    out.write_text(json.dumps(meta, indent=2))
    print(f"\nreferencia guardada en {out}")
    print("\nAccuracy esperada de la version CUDA (misma arquitectura):")
    for r in results:
        print(f"  parche {r['patch']:2d}x{r['patch']:<2d} ({r['n_tokens']:2d} tokens): "
              f"{r['test_acc']*100:.2f}%")


if __name__ == "__main__":
    main()
