#!/usr/bin/env bash
# run_go_benchmarks.sh -- Formal Go benchmarks for symbolic regression.
#
# Measures evol.Run() duration with Workers=1,2,4,8 using the Go benchmark
# framework (-benchtime=1x -count=1 per round, 5 rounds with alternating order).
# Results are written to benchmarks/results/go_bench_TIMESTAMP.txt.
#
# Usage (from repository root):
#   bash benchmarks/run_go_benchmarks.sh
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────

RESULTS_DIR="benchmarks/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
GOBENCH_OUT="$RESULTS_DIR/go_bench_${TIMESTAMP}.txt"
WORKERS_FWD=(1 2 4 8)
WORKERS_REV=(8 4 2 1)

# ── Repo root check ────────────────────────────────────────────────────────

[[ -f "go.mod" ]] || {
    echo "ERROR: Ejecutar desde la raiz del repositorio." >&2
    exit 1
}

# ── Preflight: compile test package (catches missing embed file) ───────────

echo "Verificando compilacion del paquete benchmarks/..."
if ! go test -run '^$' -bench '^$' -count=1 ./benchmarks/; then
    echo "" >&2
    echo "ERROR: El paquete benchmarks/ no compila." >&2
    echo "       Si falta el dataset, generarlo con:" >&2
    echo "         go run ./benchmarks/gen_dataset/main.go" >&2
    exit 1
fi
echo ""

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

# ── Session header (terminal + file) ──────────────────────────────────────

{
    echo "======================================================"
    echo "  Go Benchmark -- Regresion Simbolica Evolutiva"
    echo "======================================================"
    echo "Timestamp     : $TIMESTAMP"
    echo "CPU model     : $CPU_MODEL"
    echo "Kernel        : $KERNEL"
    echo "CPU governor  : $CPU_GOVERNOR"
    echo "NumCPU        : $NUM_CPU"
    echo "GOMAXPROCS    : $GOMAXPROCS_EFF  (runtime.GOMAXPROCS(0))"
    echo "Go version    : $GO_VERSION"
    echo "GOOS/GOARCH   : $GOOS/$GOARCH"
    echo "Scope         : exclusively evol.Run()"
    echo "Workers set   : ${WORKERS_FWD[*]}"
    echo "Rounds        : 1 warmup/cfg + 5 formal (alternating order)"
    echo "Flags         : -benchtime=1x -count=1 -benchmem"
    echo "======================================================"
} | tee "$GOBENCH_OUT"
echo "" | tee -a "$GOBENCH_OUT"

# ── Warmups: once per configuration, before formal rounds ─────────────────

{
    echo "-- Warmups -------------------------------------------------------"
} | tee -a "$GOBENCH_OUT"

for W in "${WORKERS_FWD[@]}"; do
    echo "[warmup] Workers=$W" | tee -a "$GOBENCH_OUT"
    go test \
        -run '^$' \
        -bench="BenchmarkRun_Workers${W}$" \
        -benchtime=1x -count=1 \
        ./benchmarks/ > /dev/null
    # set -e exits the script if go test fails
done
echo "" | tee -a "$GOBENCH_OUT"

# ── Formal rounds with alternating order ──────────────────────────────────

ROUND_ORDERS=(
    "${WORKERS_FWD[*]}"
    "${WORKERS_REV[*]}"
    "${WORKERS_FWD[*]}"
    "${WORKERS_REV[*]}"
    "${WORKERS_FWD[*]}"
)

declare -i ROUND=0

for ORDER in "${ROUND_ORDERS[@]}"; do
    ROUND=$((ROUND + 1))

    {
        echo "-- Ronda $ROUND ($ORDER) -----------------------------------------"
    } | tee -a "$GOBENCH_OUT"

    for W in $ORDER; do
        echo "[round $ROUND / workers $W]" | tee -a "$GOBENCH_OUT"
        # pipefail ensures a go test failure is not hidden by tee
        go test \
            -run '^$' \
            -bench="BenchmarkRun_Workers${W}$" \
            -benchtime=1x -count=1 \
            -benchmem \
            ./benchmarks/ | tee -a "$GOBENCH_OUT"
    done
    echo "" | tee -a "$GOBENCH_OUT"
done

# ── Summary ────────────────────────────────────────────────────────────────

{
    echo "======================================================"
    echo "  Results: $GOBENCH_OUT"
    echo "======================================================"
    echo ""
    echo "To analyze with benchstat (optional, install separately):"
    echo "  go install golang.org/x/perf/cmd/benchstat@latest"
    echo "  benchstat $GOBENCH_OUT"
    echo ""
    echo "Complementary experiment (Workers=12, run separately):"
    echo "  go test -run '^$' -bench=BenchmarkRun_Workers12 -benchtime=1x -count=5 -benchmem ./benchmarks/"
} | tee -a "$GOBENCH_OUT"

exit 0
