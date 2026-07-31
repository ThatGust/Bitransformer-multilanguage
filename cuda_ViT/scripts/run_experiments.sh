#!/usr/bin/env bash
# ============================================================================
# Ejecuta la bateria completa de experimentos y deja los CSV en results/.
#
#   Experimento 1 (tamano de parche): 3 entrenamientos, uno por tamano de
#   parche, con el kernel de atencion en memoria compartida.
#
#   Experimento 2 (memoria global vs compartida): un microbenchmark aislado de
#   los dos kernels criticos, mas 3 entrenamientos cortos con el kernel en
#   memoria global para medir tambien el impacto extremo a extremo.
#
# Uso:
#   ./scripts/run_experiments.sh                # todo
#   ./scripts/run_experiments.sh smoke          # una sola corrida de prueba
#
# IMPORTANTE: el modo 'smoke' es el que hay que correr PRIMERO. Verifica de
# punta a punta una unica configuracion y muestra el tiempo de la primera
# epoca antes de comprometerse con la bateria completa.
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

BIN=./bin/vit_train
BENCH=./bin/bench_attention
DATA=data
OUT=results

EPOCHS=${EPOCHS:-15}
TRAIN_N=${TRAIN_N:-5000}
TEST_N=${TEST_N:-2000}
BATCH=${BATCH:-64}
LR=${LR:-1e-3}

if [ ! -x "$BIN" ]; then
    echo "Falta $BIN. Ejecuta 'make' primero."
    exit 1
fi

mkdir -p "$OUT"

run_train () {
    local patch=$1 mem=$2 tag=$3
    echo ""
    echo "=============================================================="
    echo " entrenando: parche ${patch}x${patch}, atencion en memoria ${mem}"
    echo "=============================================================="
    $BIN --data-dir "$DATA" --out-dir "$OUT" --tag "$tag" \
         --patch "$patch" --attn-mem "$mem" \
         --train-n "$TRAIN_N" --test-n "$TEST_N" \
         --epochs "$EPOCHS" --batch "$BATCH" --lr "$LR"
}

if [ "${1:-all}" = "smoke" ]; then
    echo "### CORRIDA DE PRUEBA: una sola configuracion, 2 epocas ###"
    EPOCHS=2 run_train 7 shared smoke_p7_shared
    echo ""
    echo "Si el tiempo por epoca es razonable (< 30 s), lanza la bateria completa:"
    echo "    ./scripts/run_experiments.sh"
    exit 0
fi

# ---- Experimento 1: efecto del tamano de parche -----------------------------
for p in 4 7 14; do
    run_train "$p" shared "p${p}_shared"
done

# ---- Experimento 2a: microbenchmark aislado de los kernels ------------------
echo ""
echo "=============================================================="
echo " microbenchmark: memoria global vs compartida (kernels aislados)"
echo "=============================================================="
$BENCH --out-dir "$OUT" --batch "$BATCH"

# ---- Experimento 2b: impacto extremo a extremo ------------------------------
# Solo 3 epocas: aqui interesa el tiempo por epoca, no la accuracy final
# (ambas versiones calculan exactamente lo mismo y convergen igual).
for p in 4 7 14; do
    echo ""
    echo "=============================================================="
    echo " tiempo extremo a extremo con memoria GLOBAL, parche ${p}x${p}"
    echo "=============================================================="
    $BIN --data-dir "$DATA" --out-dir "$OUT" --tag "p${p}_global" \
         --patch "$p" --attn-mem global \
         --train-n "$TRAIN_N" --test-n "$TEST_N" \
         --epochs 3 --batch "$BATCH" --lr "$LR"
done

echo ""
echo "Listo. CSV generados en $OUT/:"
ls -1 "$OUT"
echo ""
echo "Siguiente paso: python3 scripts/make_report_assets.py"
