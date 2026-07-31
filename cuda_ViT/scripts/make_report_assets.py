#!/usr/bin/env python3
"""
make_report_assets.py -- Convierte los CSV de results/ en las figuras y tablas
que consume informe/informe.tex.

    python3 scripts/make_report_assets.py

Lee:
    results/summary_p{4,7,14}_shared.csv   (experimento 1)
    results/summary_p{4,7,14}_global.csv   (experimento 2, extremo a extremo)
    results/history_*.csv                  (curvas por epoca)
    results/bench_attention.csv            (experimento 2, kernels aislados)

Escribe:
    informe/figs/fig_patch_tradeoff.png    accuracy y tiempo por tamano de parche
    informe/figs/fig_curves.png            curvas de perdida y accuracy por epoca
    informe/figs/fig_memory.png            global vs compartida y speedup
    informe/tables/tab_patch.tex           tabla del experimento 1
    informe/tables/tab_bench.tex           tabla del experimento 2

Si falta algun CSV, genera un marcador de posicion en su lugar y avisa, de modo
que el informe SIEMPRE compile aunque todavia no se hayan corrido los
experimentos en Colab.
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "informe" / "figs"
TABLES = ROOT / "informe" / "tables"

# Paleta categorica validada (separacion garantizada para daltonismo deutan y
# tritan; ver scripts/validate_palette.js del sistema de diseno). El orden es
# fijo: parche 4 -> slot 1, parche 7 -> slot 2, parche 14 -> slot 3.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
PATCH_COLOR = {4: C1, 7: C2, 14: C3}
MEM_COLOR = {"global": C2, "shared": C1}

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dedddb"

PATCHES = [4, 7, 14]

# Los nombres de kernel en el CSV son ASCII puro; aqui se traducen a notacion
# matematica legible para las figuras.
KERNEL_LABEL = {"QK^T": "$QK^{T}$", "PxV": r"$P \cdot V$"}
KERNEL_TEX = {"QK^T": r"$QK^{T}$", "PxV": r"$P \cdot V$"}


def style():
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
    })


def clean(ax, ygrid=True):
    """Ejes recesivos: sin marcos superiores/derechos, rejilla tenue detras."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    if ygrid:
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
        ax.xaxis.grid(False)


def read_csv(path):
    if not path.exists():
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows or None


def placeholder(path, msg):
    """Figura de reemplazo cuando faltan los datos, para que el .tex compile."""
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.axis("off")
    ax.text(0.5, 0.55, "figura pendiente", ha="center", va="center",
            fontsize=13, color=INK)
    ax.text(0.5, 0.28, msg, ha="center", va="center", fontsize=8.5, color=INK2)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [pendiente] {path.name}: {msg}")


def bar_labels(ax, bars, fmt="{:.1f}", dy=0.01):
    """Etiquetas directas sobre cada barra.
    Obligatorias, no decorativas: sostienen la identidad de la serie cuando el
    contraste del color contra el fondo blanco queda por debajo de 3:1."""
    top = max(b.get_height() for b in bars) if bars else 1.0
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + top * dy,
                fmt.format(b.get_height()), ha="center", va="bottom",
                fontsize=8, color=INK)


# --------------------------------------------------------------------------
# Figura 1: compromiso entre numero de parches, accuracy y tiempo
# --------------------------------------------------------------------------
def fig_patch_tradeoff(summaries):
    path = FIGS / "fig_patch_tradeoff.png"
    if not summaries:
        placeholder(path, "corre scripts/run_experiments.sh para generar results/summary_p*_shared.csv")
        return

    ps = sorted(summaries.keys())
    labels = [f"{p}x{p}\n{summaries[p]['n_tokens']} tokens" for p in ps]
    colors = [PATCH_COLOR[p] for p in ps]
    accs = [float(summaries[p]["test_acc"]) * 100 for p in ps]
    eps = [float(summaries[p]["epoch_sec_mean"]) for p in ps]
    inf = [float(summaries[p]["infer_ms_per_batch"]) for p in ps]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))

    b = axes[0].bar(labels, accs, color=colors, width=0.6)
    axes[0].set_title("Accuracy en test")
    axes[0].set_ylabel("%")
    axes[0].set_ylim(0, max(accs) * 1.22)
    bar_labels(axes[0], b, "{:.1f}")

    b = axes[1].bar(labels, eps, color=colors, width=0.6)
    axes[1].set_title("Tiempo por epoca")
    axes[1].set_ylabel("segundos")
    axes[1].set_ylim(0, max(eps) * 1.22)
    bar_labels(axes[1], b, "{:.2f}")

    b = axes[2].bar(labels, inf, color=colors, width=0.6)
    axes[2].set_title("Inferencia por batch")
    axes[2].set_ylabel("ms")
    axes[2].set_ylim(0, max(inf) * 1.22)
    bar_labels(axes[2], b, "{:.2f}")

    for ax in axes:
        clean(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  [ok] {path.name}")


# --------------------------------------------------------------------------
# Figura 2: curvas de aprendizaje
# --------------------------------------------------------------------------
def fig_curves(histories):
    path = FIGS / "fig_curves.png"
    if not histories:
        placeholder(path, "corre scripts/run_experiments.sh para generar results/history_p*_shared.csv")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for p in sorted(histories.keys()):
        rows = histories[p]
        ep = [int(r["epoch"]) for r in rows]
        lbl = f"parche {p}x{p}"
        axes[0].plot(ep, [float(r["train_loss"]) for r in rows],
                     color=PATCH_COLOR[p], linewidth=2, label=lbl)
        axes[0].plot(ep, [float(r["test_loss"]) for r in rows],
                     color=PATCH_COLOR[p], linewidth=1.4, linestyle="--", alpha=0.75)
        axes[1].plot(ep, [float(r["test_acc"]) * 100 for r in rows],
                     color=PATCH_COLOR[p], linewidth=2, label=lbl)

    axes[0].set_title("Perdida (solida: train, punteada: test)")
    axes[0].set_xlabel("epoca")
    axes[0].set_ylabel("entropia cruzada")
    axes[1].set_title("Accuracy en test")
    axes[1].set_xlabel("epoca")
    axes[1].set_ylabel("%")
    for ax in axes:
        clean(ax)
    # La leyenda va en el panel de accuracy, cuyo cuadrante inferior derecho
    # queda vacio: en el de perdida chocaria con las curvas de test.
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  [ok] {path.name}")


# --------------------------------------------------------------------------
# Figura 3: memoria global vs compartida
# --------------------------------------------------------------------------
def fig_memory(bench):
    path = FIGS / "fig_memory.png"
    if not bench:
        placeholder(path, "corre bin/bench_attention para generar results/bench_attention.csv")
        return

    kernels = []
    for r in bench:
        if r["kernel"] not in kernels:
            kernels.append(r["kernel"])

    fig, axes = plt.subplots(1, len(kernels) + 1, figsize=(7.4, 2.7))

    for ax, kern in zip(axes, kernels):
        rows = sorted((r for r in bench if r["kernel"] == kern),
                      key=lambda r: int(r["n_tokens"]))
        x = range(len(rows))
        w = 0.38
        gl = [float(r["ms_global"]) * 1000 for r in rows]   # a microsegundos
        sh = [float(r["ms_shared"]) * 1000 for r in rows]
        # Separacion de 2 px entre barras adyacentes mediante el ancho y el
        # desplazamiento; el fondo blanco hace de espaciador.
        b1 = ax.bar([i - w / 2 - 0.01 for i in x], gl, width=w,
                    color=MEM_COLOR["global"], label="global")
        b2 = ax.bar([i + w / 2 + 0.01 for i in x], sh, width=w,
                    color=MEM_COLOR["shared"], label="compartida")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"T={r['n_tokens']}" for r in rows])
        ax.set_title(f"{KERNEL_LABEL.get(kern, kern)}: tiempo de kernel")
        ax.set_ylabel("microsegundos")
        ax.set_ylim(0, max(gl + sh) * 1.30)
        bar_labels(ax, list(b1) + list(b2), "{:.0f}")
        clean(ax)
    axes[0].legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))

    # Panel de speedup: una linea por kernel frente al numero de tokens.
    ax = axes[-1]
    for kern, col in zip(kernels, (C1, C2, C3)):
        rows = sorted((r for r in bench if r["kernel"] == kern),
                      key=lambda r: int(r["n_tokens"]))
        xs = [int(r["n_tokens"]) for r in rows]
        ys = [float(r["speedup"]) for r in rows]
        # markeredgecolor blanco: si dos kernels dan speedups casi iguales, el
        # anillo del color de fondo mantiene los marcadores distinguibles al
        # solaparse.
        ax.plot(xs, ys, marker="o", markersize=6, linewidth=2, color=col,
                markeredgecolor="white", markeredgewidth=1.2,
                label=KERNEL_LABEL.get(kern, kern))
        for xv, yv in zip(xs, ys):
            ax.annotate(f"{yv:.2f}x", (xv, yv), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8, color=INK)
    ax.axhline(1.0, color=GRID, linewidth=1.2, zorder=0)
    ax.set_title("Speedup compartida / global")
    ax.set_xlabel("tokens (T)")
    ax.set_ylabel("factor")
    lo, hi = min(min(float(r["speedup"]) for r in bench), 1.0), max(float(r["speedup"]) for r in bench)
    ax.set_ylim(lo - 0.08 * (hi - lo + 0.1), hi + 0.22 * (hi - lo + 0.1))
    clean(ax)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  [ok] {path.name}")


# --------------------------------------------------------------------------
# Tablas LaTeX (booktabs)
# --------------------------------------------------------------------------
def tab_patch(summaries, globals_):
    path = TABLES / "tab_patch.tex"
    if not summaries:
        path.write_text("\\textit{Tabla pendiente: corre scripts/run\\_experiments.sh.}\n")
        print(f"  [pendiente] {path.name}")
        return

    lines = [
        "% Generado por scripts/make_report_assets.py -- no editar a mano",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Parche & Tokens & Parametros & Acc. test & Total & s/epoca & Infer. & Mem. GPU \\\\",
        " & & & (\\%) & (s) & (s) & (ms/batch) & (MiB) \\\\",
        "\\midrule",
    ]
    for p in sorted(summaries):
        s = summaries[p]
        lines.append(
            f"${p}\\times{p}$ & {s['n_tokens']} & {int(s['params']):,} & "
            f"{float(s['test_acc'])*100:.2f} & {float(s['train_total_sec']):.1f} & "
            f"{float(s['epoch_sec_mean']):.2f} & {float(s['infer_ms_per_batch']):.2f} & "
            f"{float(s['gpu_mem_mib']):.0f} \\\\".replace(",", "\\,")
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"  [ok] {path.name}")


def tab_bench(bench, summaries, globals_):
    path = TABLES / "tab_bench.tex"
    if not bench:
        path.write_text("\\textit{Tabla pendiente: corre bin/bench\\_attention.}\n")
        print(f"  [pendiente] {path.name}")
        return

    lines = [
        "% Generado por scripts/make_report_assets.py -- no editar a mano",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Kernel & Parche & $T$ & Global & Compartida & Speedup & GFLOP/s \\\\",
        " & & & ($\\mu$s) & ($\\mu$s) & & (compartida) \\\\",
        "\\midrule",
    ]
    for r in sorted(bench, key=lambda r: (r["kernel"], -int(r["n_tokens"]))):
        p = int(r["patch"])
        lines.append(
            f"{KERNEL_TEX.get(r['kernel'], r['kernel'])} & "
            f"${p}\\times{p}$ & {r['n_tokens']} & "
            f"{float(r['ms_global'])*1000:.1f} & {float(r['ms_shared'])*1000:.1f} & "
            f"{float(r['speedup']):.2f}$\\times$ & {float(r['gflops_shared']):.1f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"  [ok] {path.name}")


def kpis(summaries, bench, globals_):
    """Genera informe/tables/kpi.tex con las cifras clave como comandos LaTeX.

    Asi la PROSA del informe (no solo las tablas) queda ligada a los CSV: al
    reentrenar y volver a correr este script, los numeros citados en el texto
    se actualizan solos. informe.tex declara valores por defecto con
    \\providecommand, de modo que compila aunque este archivo no exista.
    """
    path = TABLES / "kpi.tex"
    out = ["% Generado por scripts/make_report_assets.py -- no editar a mano"]

    def cmd(name, value):
        out.append(f"\\renewcommand{{\\{name}}}{{{value}}}")

    if summaries:
        best = max(summaries, key=lambda p: float(summaries[p]["test_acc"]))
        worst = min(summaries, key=lambda p: float(summaries[p]["test_acc"]))
        cmd("KpiBestPatch", f"${best}\\times{best}$")
        cmd("KpiBestTokens", summaries[best]["n_tokens"])
        cmd("KpiBestAcc", f"{float(summaries[best]['test_acc'])*100:.2f}")
        cmd("KpiWorstPatch", f"${worst}\\times{worst}$")
        cmd("KpiWorstTokens", summaries[worst]["n_tokens"])
        cmd("KpiWorstAcc", f"{float(summaries[worst]['test_acc'])*100:.2f}")
        cmd("KpiAccGap", f"{(float(summaries[best]['test_acc'])-float(summaries[worst]['test_acc']))*100:.2f}")
        slow = max(summaries, key=lambda p: float(summaries[p]["epoch_sec_mean"]))
        fast = min(summaries, key=lambda p: float(summaries[p]["epoch_sec_mean"]))
        ratio = float(summaries[slow]["epoch_sec_mean"]) / float(summaries[fast]["epoch_sec_mean"])
        cmd("KpiEpochRatio", f"{ratio:.1f}")
        cmd("KpiTotalTrainSec", f"{sum(float(s['train_total_sec']) for s in summaries.values()):.0f}")
        cmd("KpiParams", f"{int(summaries[min(summaries)]['params']):,}".replace(",", "\\,"))

    if bench:
        sp = {(r["kernel"], int(r["n_tokens"])): float(r["speedup"]) for r in bench}
        cmd("KpiSpeedupMax", f"{max(sp.values()):.2f}")
        cmd("KpiSpeedupMin", f"{min(sp.values()):.2f}")
        tmax = max(k[1] for k in sp)
        tmin = min(k[1] for k in sp)
        cmd("KpiSpeedupTokensMax", tmax)
        cmd("KpiSpeedupTokensMin", tmin)
        cmd("KpiSpeedupAtMax", f"{max(v for k, v in sp.items() if k[1] == tmax):.2f}")
        cmd("KpiSpeedupAtMin", f"{max(v for k, v in sp.items() if k[1] == tmin):.2f}")
        cmd("KpiBenchDiff", max(float(r["max_abs_diff"]) for r in bench))

        # Rango por kernel (no solo el global): la prosa del informe discute el
        # rango de P.V especificamente, y escribirlo a mano se desincroniza en
        # cuanto cambia un dato (paso justo eso: un valor quedo fuera del rango
        # tipeado a mano). Con macros por kernel no puede volver a pasar.
        for kern in sorted(set(k[0] for k in sp)):
            vals = [v for k, v in sp.items() if k[0] == kern]
            slug = "QKT" if kern == "QK^T" else "PxV"
            cmd(f"Kpi{slug}SpeedupMin", f"{min(vals):.2f}")
            cmd(f"Kpi{slug}SpeedupMax", f"{max(vals):.2f}")
            # tokens en el que cae el minimo de ESTE kernel (para poder decir
            # "el minimo ocurre en T=.." sin adivinar si es monotono)
            tmin_k = min((k[1] for k in sp if k[0] == kern), key=lambda t: sp[(kern, t)])
            cmd(f"Kpi{slug}SpeedupMinTokens", tmin_k)

    if summaries and globals_:
        common = sorted(set(summaries) & set(globals_))
        if common:
            r = [float(globals_[p]["epoch_sec_mean"]) / float(summaries[p]["epoch_sec_mean"])
                 for p in common]
            # Sin digitos en el nombre: LaTeX no los admite en \newcommand.
            cmd("KpiEndToEndSpeedup", f"{max(r):.2f}")

    path.write_text("\n".join(out) + "\n")
    print(f"  [ok] {path.name}")


def main():
    style()
    FIGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    summaries, histories, globals_ = {}, {}, {}
    for p in PATCHES:
        s = read_csv(RESULTS / f"summary_p{p}_shared.csv")
        if s:
            summaries[p] = s[0]
        h = read_csv(RESULTS / f"history_p{p}_shared.csv")
        if h:
            histories[p] = h
        g = read_csv(RESULTS / f"summary_p{p}_global.csv")
        if g:
            globals_[p] = g[0]

    bench = read_csv(RESULTS / "bench_attention.csv")

    print("generando figuras y tablas del informe...")
    fig_patch_tradeoff(summaries)
    fig_curves(histories)
    fig_memory(bench)
    tab_patch(summaries, globals_)
    tab_bench(bench, summaries, globals_)
    kpis(summaries, bench, globals_)

    missing = []
    if not summaries:
        missing.append("entrenamientos (summary_p*_shared.csv)")
    if not bench:
        missing.append("microbenchmark (bench_attention.csv)")
    if missing:
        print("\nFaltan datos de: " + ", ".join(missing))
        print("El informe compila igual, con marcadores en lugar de las figuras.")
        return 0

    print("\nTodo listo. Compila el informe con:  tectonic informe/informe.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
