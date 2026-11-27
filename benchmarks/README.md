# Benchmarks -- Regresion Simbolica Evolutiva -- Fase 16

Mide el efecto del numero de workers sobre el rendimiento del worker pool.
La unica variable que cambia entre corridas es `Workers`.

## Dataset

| Parametro       | Valor                                                     |
|----------------|-----------------------------------------------------------|
| Funcion        | f(x) = x^2 + 3x + 2                                      |
| Dominio        | x en [1, 20], uniformes                                   |
| Puntos totales | 625                                                       |
| Training       | 500 pts (80%)                                             |
| Validation     | 125 pts (20%)                                             |
| Archivo        | `benchmarks/datasets/benchmark_quadratic_625pts.csv`      |

## Configuracion experimental

Base: `config.BENCHMARKConfig()` -> `BALANCEDConfig()`.
Sobreescrituras especificas del experimento:

| Parametro   | Valor   | Origen      |
|------------|---------|-------------|
| Seed GA    | 42      | experimento |
| Seed gen   | 43      | experimento |
| Generations | 100    | experimento |
| Workers    | 1/2/4/8 | variable    |

Parametros heredados de BENCHMARKConfig (sin duplicar):
Population=1000, ElitismRate=5%, MutationRate=10%, CrossoverRate=80%,
Lambda=0.01, TournamentSize=3, MaxDepth=7, MaxNodes=75,
TargetMSEEnabled=false, StagnationLimit=0.

## Evaluaciones por Run() -- verificacion

```
InitPopulation:        1000 * 500      =    500 000
NextGeneration (x100): 100 * 950 * 500 = 47 500 000
finalize (val):        1 * 125         =        125
---------------------------------------------------
Total (totalPointEvals):               = 48 000 125
```

EliteCount = int(0.05 * 1000) = 50. Los elites se copian sin re-evaluar.
finalize() se llama exactamente una vez al final de Run().

## Formulas

**Throughput** (Go benchmarks, alcance = Run()):
```
Meval/s = 48 000 125 / elapsed_Run_s / 1 000 000
```

**Speedup:** `Speedup(n) = Mediana(T1) / Mediana(Tn)`

**Efficiency:** `Efficiency(n) = Speedup(n) / n`  (sin truncamiento de rango)

**CPU utilization** (CLI, alcance = proceso completo):
```
CPU cores used = (user_s + sys_s) / real_s
CPU util %     = (user_s + sys_s) / real_s / GOMAXPROCS_effective * 100
```

Nota: `%CPU` de GNU time puede superar 100% (referencia = 1 nucleo logico).

## Workers=1 como baseline

Workers=1 opera con la infraestructura completa del worker pool (canal jobs,
canal results, una goroutine worker). No es una implementacion secuencial sin pool.
El proyecto no cuenta con una ruta de evaluacion alternativa sin pool.
Speedup se calcula respecto de Workers=1 con esta precision documentada.

## Consistencia de semillas

Ambos metodos usan Seed GA=42 y GenConfig.Seed=43:
- Go benchmarks: `buildBenchConfig()` asigna explicitamente ambas seeds.
- CLI: `--seed 42` -> `main.go` asigna `cfg.GenConfig.Seed = 42 + 1 = 43`.

## Prerequisitos

**Paso 0 -- Generar el dataset (una sola vez, luego commitear):**
```bash
go run ./benchmarks/gen_dataset/main.go
git add benchmarks/datasets/benchmark_quadratic_625pts.csv
git commit -m "bench: add benchmark dataset (625 pts, x^2+3x+2)"
```

**Paso 1 -- Verificar que todo compila:**
```bash
go test ./... -count=1
```

**Paso 2 -- Compilar el binario de produccion (para run_benchmark.sh):**
```bash
go build -o symbolic-regression ./cmd/symbolic-regression
```

## Ejecucion

### Go benchmarks (miden exclusivamente Run())

```bash
bash benchmarks/run_go_benchmarks.sh
```

Genera: `benchmarks/results/go_bench_TIMESTAMP.txt`

### CLI end-to-end (miden el proceso completo)

```bash
bash benchmarks/run_benchmark.sh
```

Genera: `benchmarks/results/cli_TIMESTAMP.csv`

### Experimento complementario Workers=12

```bash
go test \
    -run '^$' \
    -bench=BenchmarkRun_Workers12 \
    -benchtime=1x -count=5 -benchmem \
    ./benchmarks/ \
    | tee "benchmarks/results/go_bench_w12_$(date +%Y%m%d_%H%M%S).txt"
```

Separado del conjunto formal (Workers=1,2,4,8). Especifico para el i7-1255U.

### Analisis estadistico (opcional, sin agregar a go.mod)

```bash
go install golang.org/x/perf/cmd/benchstat@latest
benchstat benchmarks/results/go_bench_*.txt
```

## Separacion de tablas de resultados

**Go benchmarks:** ns/op, Meval/s, B/op, allocs/op -- alcance = Run()
**CLI end-to-end:** real, user, sys, CPU cores, CPU util % -- alcance = proceso completo

No mezclar tiempos ni throughputs entre ambas fuentes.

## Workers=12 (experimento complementario)

El i7-1255U tiene 12 CPUs logicos (2 P-cores x HT + 8 E-cores).
Los resultados se documentan en una seccion separada del informe final.
No forman parte del conjunto formal.

## Limitaciones

- Workers=1 incluye overhead del canal del worker pool (no es secuencial puro).
- Los P-cores y E-cores del i7-1255U tienen rendimiento diferente.
- Throttling termico puede afectar las rondas finales; la rotacion mitiga el sesgo.
- La eficiencia paralela depende de la fraccion de tiempo en evaluatePopulation.
