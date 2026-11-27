#!/usr/bin/env bash
# run_benchmark.sh -- CLI end-to-end benchmark for symbolic regression.
#
# Measures elapsed time and CPU utilization for Workers=1,2,4,8 using
# /usr/bin/time. Results are written to benchmarks/results/.
#
# Usage (from repository root):
#   1. go build -o symbolic-regression ./cmd/symbolic-regression
#   2. bash benchmarks/run_benchmark.sh
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────

BINARY="./symbolic-regression"
DATASET="benchmarks/datasets/benchmark_quadratic_625pts.csv"
RESULTS_DIR="benchmarks/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SEED=42
GENERATIONS=100
PROFILE="benchmark"
WORKERS_FWD=(1 2 4 8)
WORKERS_REV=(8 4 2 1)
TIME_CMD="/usr/bin/time"

# ── Repo root check ────────────────────────────────────────────────────────

[[ -f "go.mod" ]] || {
    echo "ERROR: Ejecutar desde la raiz del repositorio." >&2
    exit 1
}

# ── Prerequisite checks ────────────────────────────────────────────────────

if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: Compilar primero:" >&2
    echo "  go build -o symbolic-regression ./cmd/symbolic-regression" >&2
    exit 1
fi

if [[ ! -f "$DATASET" ]]; then
    echo "ERROR: Dataset no encontrado: $DATASET" >&2
    echo "  Generarlo con: go run ./benchmarks/gen_dataset/main.go" >&2
    exit 1
fi

if ! command -v "$TIME_CMD" &>/dev/null; then
    echo "ERROR: $TIME_CMD no encontrado." >&2
    exit 1
fi

# ── Create RESULTS_DIR before sysinfo compilation ─────────────────────────

mkdir -p "$RESULTS_DIR"

# ── GOMAXPROCS effective (temp binary with guaranteed cleanup) ─────────────

SYSINFO_BIN=$(mktemp)
trap 'rm -f "$SYSINFO_BIN"' EXIT

go build -o "$SYSINFO_BIN" ./benchmarks/sysinfo/

SYSINFO_OUT=$("$SYSINFO_BIN")
GOMAXPROCS_EFF=$(echo "$SYSINFO_OUT" | awk -F= '/^GOMAXPROCS/{print $2}')
NUM_CPU=$(echo       "$SYSINFO_OUT" | awk -F= '/^NumCPU/{print $2}')
GO_VERSION=$(echo    "$SYSINFO_OUT" | awk -F= '/^GoVersion/{print $2}')
GOOS=$(echo          "$SYSINFO_OUT" | awk -F= '/^GOOS/{print $2}')
GOARCH=$(echo        "$SYSINFO_OUT" | awk -F= '/^GOARCH/{print $2}')

if ! [[ "$GOMAXPROCS_EFF" =~ ^[0-9]+$ ]] || [[ "$GOMAXPROCS_EFF" -le 0 ]]; then
    echo "ERROR: No se pudo obtener GOMAXPROCS efectivo (valor: '$GOMAXPROCS_EFF')." >&2
    exit 1
fi

# ── System info from OS ────────────────────────────────────────────────────

KERNEL=$(uname -r)
CPU_MODEL=$(awk -F: '/^model name[[:space:]]*:/ {
    sub(/^[[:space:]]*/, "", $2)
    print $2
    exit
}' /proc/cpuinfo 2>/dev/null || true)
CPU_MODEL=${CPU_MODEL:-unavailable}
GOVERNOR_PATH="/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
if [[ -r "$GOVERNOR_PATH" ]]; then
    CPU_GOVERNOR=$(cat "$GOVERNOR_PATH")
else
    CPU_GOVERNOR="unavailable"
fi

# ── Session header ─────────────────────────────────────────────────────────

ENV_FILE="$RESULTS_DIR/environment_${TIMESTAMP}.txt"
{
    echo "======================================================"
    echo "  CLI Benchmark -- Regresion Simbolica Evolutiva"
    echo "======================================================"
    echo "Timestamp     : $TIMESTAMP"
    echo "CPU model     : $CPU_MODEL"
    echo "Kernel        : $KERNEL"
    echo "CPU governor  : $CPU_GOVERNOR"
    echo "NumCPU        : $NUM_CPU"
    echo "GOMAXPROCS    : $GOMAXPROCS_EFF  (runtime.GOMAXPROCS(0))"
    echo "Go version    : $GO_VERSION"
    echo "GOOS/GOARCH   : $GOOS/$GOARCH"
    echo "Dataset       : $DATASET"
    echo "Profile       : $PROFILE | Seed: $SEED | Generations: $GENERATIONS"
    echo "Workers set   : ${WORKERS_FWD[*]}"
    echo "Rounds        : 1 warmup/cfg + 5 formal (alternating order)"
    echo "======================================================"
} | tee "$ENV_FILE"
echo ""

# ── CSV header ─────────────────────────────────────────────────────────────

CSV_OUT="$RESULTS_DIR/cli_${TIMESTAMP}.csv"
echo "round,workers,real_s,user_s,sys_s,cpu_cores_used,cpu_util_pct,exit_code" \
    > "$CSV_OUT"

# ── Auxiliary function ─────────────────────────────────────────────────────

# run_one WORKERS OUT_DIR
# Separates three output streams:
#   stdout       -> stdout.txt (program output)
#   stderr       -> stderr.txt (program errors)
#   time metrics -> time.txt   (/usr/bin/time -o flag)
run_one() {
    local workers="$1"
    local out_dir="$2"
    mkdir -p "$out_dir"
    "$TIME_CMD" \
        -f "%e %U %S" \
        -o "$out_dir/time.txt" \
        "$BINARY" \
            --data        "$DATASET" \
            --profile     "$PROFILE" \
            --seed        "$SEED" \
            --workers     "$workers" \
            --generations "$GENERATIONS" \
        > "$out_dir/stdout.txt" \
        2> "$out_dir/stderr.txt"
}

# ── Phase 1: warmup runs (once per configuration) ─────────────────────────

echo "-- Warmups -------------------------------------------------------"
for W in "${WORKERS_FWD[@]}"; do
    echo "  [warmup] Workers=$W..."
    run_one "$W" "$RESULTS_DIR/warmup_w${W}_${TIMESTAMP}"
    echo "  [warmup] Workers=$W completado."
done
echo ""

# ── Phase 2: formal rounds with alternating order ──────────────────────────

ROUND_ORDERS=(
    "${WORKERS_FWD[*]}"
    "${WORKERS_REV[*]}"
    "${WORKERS_FWD[*]}"
    "${WORKERS_REV[*]}"
    "${WORKERS_FWD[*]}"
)

declare -i INCOMPLETE=0
declare -i ROUND=0

for ORDER in "${ROUND_ORDERS[@]}"; do
    ROUND=$((ROUND + 1))
    echo "-- Ronda $ROUND ($ORDER) -----------------------------------------"

    for W in $ORDER; do
        echo "  [round $ROUND / workers $W]..."

        OUT_DIR="$RESULTS_DIR/run_r${ROUND}_w${W}_${TIMESTAMP}"

        set +e
        run_one "$W" "$OUT_DIR"
        EXIT_CODE=$?
        set -e

        if [[ $EXIT_CODE -ne 0 ]]; then
            echo "  ERROR: round $ROUND workers $W failed (exit $EXIT_CODE) -- not recorded." >&2
            INCOMPLETE=1
            continue
        fi

        TIME_LINE=$(cat "$OUT_DIR/time.txt")
        REAL_S=$(echo "$TIME_LINE" | awk '{print $1}')
        USER_S=$(echo "$TIME_LINE" | awk '{print $2}')
        SYS_S=$(echo  "$TIME_LINE" | awk '{print $3}')

        CPU_CORES=$(LC_ALL=C awk "BEGIN {printf \"%.3f\", ($USER_S + $SYS_S) / $REAL_S}")
        CPU_UTIL=$(LC_ALL=C awk "BEGIN {printf \"%.2f\", \
    ($USER_S + $SYS_S) / $REAL_S / $GOMAXPROCS_EFF * 100}")

        echo "$ROUND,$W,$REAL_S,$USER_S,$SYS_S,$CPU_CORES,$CPU_UTIL,$EXIT_CODE" \
            >> "$CSV_OUT"
        echo "    real=${REAL_S}s  user=${USER_S}s  sys=${SYS_S}s" \
             "  cores=${CPU_CORES}  util=${CPU_UTIL}%"
    done
    echo ""
done

# ── Summary ────────────────────────────────────────────────────────────────

echo "======================================================"
echo "  Entorno:   $ENV_FILE"
echo "  CSV:       $CSV_OUT"
echo "  Logs:      $RESULTS_DIR/run_r*_${TIMESTAMP}/"
echo "======================================================"

if [[ $INCOMPLETE -ne 0 ]]; then
    echo "" >&2
    echo "ERROR: Benchmark finalizado con resultados incompletos." >&2
    echo "       Revisar logs en: $RESULTS_DIR/run_r*_${TIMESTAMP}/" >&2
    exit 1
fi

exit 0
