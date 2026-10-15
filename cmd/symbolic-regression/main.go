// Regresión Simbólica Evolutiva — Fase 11: perfiles de ejecución.
//
// El programa carga un dataset CSV, lo divide en training/validation,
// ejecuta el algoritmo genético con el perfil elegido e imprime el resultado.
// No recibe la fórmula original; únicamente los datos (x, y).
//
// Uso:
//
//	go run ./cmd/symbolic-regression [--profile PERFIL] [--seed N] [DATASET]
//
// Ejemplos:
//
//	go run ./cmd/symbolic-regression --profile fast datasets/linear.csv
//	go run ./cmd/symbolic-regression --profile balanced --seed 7 datasets/linear.csv
//	go run ./cmd/symbolic-regression datasets/linear.csv          # balanced + seed 42
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/LautiSeverino/symbolic-regression/internal/config"
	"github.com/LautiSeverino/symbolic-regression/internal/dataset"
	"github.com/LautiSeverino/symbolic-regression/internal/expression"
	"github.com/LautiSeverino/symbolic-regression/internal/fitness"
	"github.com/LautiSeverino/symbolic-regression/internal/genetic"
)

const (
	defaultDataset = "datasets/linear.csv"
	trainRatio     = 0.80
	printEvery     = 10
)

func main() {
	// ── Flags ─────────────────────────────────────────────────────────────
	profileFlag := flag.String("profile", "balanced",
		"perfil de ejecución: fast | balanced | intensive | benchmark")
	seedFlag := flag.Int64("seed", 42,
		"semilla para reproducibilidad (misma seed + mismo perfil + mismo dataset = mismo resultado)")
	flag.Parse()

	// Detectar si --seed fue pasado explícitamente (relevante para BENCHMARK).
	seedExplicit := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "seed" {
			seedExplicit = true
		}
	})

	// El dataset se pasa como argumento posicional (igual que en Fase 10).
	dataPath := defaultDataset
	if args := flag.Args(); len(args) > 0 {
		dataPath = args[0]
	}

	// ── Cargar perfil ──────────────────────────────────────────────────────
	cfg, err := config.Parse(*profileFlag)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	// Advertencia específica para BENCHMARK sin seed explícita.
	// BENCHMARK está pensado para medir y comparar; sin seed fija los resultados
	// no son comparables entre ejecuciones distintas.
	if strings.EqualFold(strings.TrimSpace(*profileFlag), "benchmark") && !seedExplicit {
		fmt.Fprintln(os.Stderr,
			"aviso: perfil benchmark usado sin --seed explícito; "+
				"los resultados pueden no ser comparables entre ejecuciones")
	}

	// Asignar seeds: el GA y el generador usan fuentes independientes.
	cfg.Seed = *seedFlag
	cfg.GenConfig.Seed = *seedFlag + 1

	fmt.Println("══════════════════════════════════════════")
	fmt.Println("  Regresión Simbólica Evolutiva — Fase 11")
	fmt.Println("══════════════════════════════════════════")
	fmt.Println()

	// ── 1. Cargar dataset ──────────────────────────────────────────────────
	ds, err := dataset.LoadCSV(dataPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error al cargar %q: %v\n", dataPath, err)
		os.Exit(1)
	}
	fmt.Printf("Dataset  : %s  (%d puntos)\n", dataPath, ds.Len())

	// ── 2. División train / validation ────────────────────────────────────
	train, val, err := dataset.Split(ds, trainRatio, *seedFlag)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error al dividir dataset: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Split    : %.0f%% train = %d puntos | %.0f%% val = %d puntos\n",
		trainRatio*100, train.Len(), (1-trainRatio)*100, val.Len())
	fmt.Println()

	// ── 3. Información de configuración ───────────────────────────────────
	fmt.Printf("Profile  : %s\n", strings.ToLower(*profileFlag))
	fmt.Printf("Config   : pop=%d | gen=%d | depth=%d | nodes=%d | workers=%d | seed=%d\n",
		cfg.PopulationSize, cfg.Generations,
		cfg.GenConfig.MaxDepth, cfg.GenConfig.MaxNodes, cfg.Workers, *seedFlag)
	fmt.Printf("Fitness  : MSE + λ·complexity  (λ=%.3f)\n", cfg.Lambda)
	fmt.Println()

	fmt.Printf("%-12s  %-14s  %-14s  %-8s  %s\n",
		"Generation", "MSE", "Fitness", "Nodes", "Expression")
	fmt.Println("──────────────────────────────────────────────────────────────────────")

	// ── 4. Callback de progreso ───────────────────────────────────────────
	start := time.Now()
	cfg.OnGeneration = func(gen int, best genetic.Individual) {
		if gen == 0 || gen%printEvery == 0 || gen == cfg.Generations {
			fmt.Printf("%-12d  %-14.6f  %-14.6f  %-8d  %s\n",
				gen,
				best.Result.MSE,
				best.Result.Fitness,
				best.Result.Complexity,
				expression.Format(best.Expr),
			)
		}
	}

	// ── 5. Crear y ejecutar la evolución ──────────────────────────────────
	evol, err := genetic.NewEvolution(cfg, train)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error al inicializar evolución: %v\n", err)
		os.Exit(1)
	}

	best, err := evol.Run()
	if err != nil {
		fmt.Fprintf(os.Stderr, "error durante la evolución: %v\n", err)
		os.Exit(1)
	}
	elapsed := time.Since(start)

	// ── 6. Evaluar en validation ──────────────────────────────────────────
	valResult, err := fitness.Evaluate(best.Expr, val, 0)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error al evaluar en validation: %v\n", err)
		os.Exit(1)
	}

	// ── 7. Resultado final ────────────────────────────────────────────────
	totalEvals := cfg.PopulationSize * cfg.Generations * train.Len()
	fmt.Println()
	fmt.Println("══════════════════════════════════════════")
	fmt.Println("  RESULT")
	fmt.Println("══════════════════════════════════════════")
	fmt.Println()
	fmt.Printf("Expression     : %s\n", expression.Format(best.Expr))
	fmt.Println()
	fmt.Printf("Training MSE   : %.10f\n", best.Result.MSE)
	fmt.Printf("Validation MSE : %.10f\n", valResult.MSE)
	fmt.Println()
	fmt.Printf("Complexity     : %d nodes\n", best.Result.Complexity)
	fmt.Println()
	fmt.Printf("Generations    : %d\n", cfg.Generations)
	fmt.Printf("Population     : %d\n", cfg.PopulationSize)
	fmt.Printf("Total evals    : %d\n", totalEvals)
	fmt.Printf("Workers        : %d\n", cfg.Workers)
	fmt.Printf("Profile        : %s\n", strings.ToLower(*profileFlag))
	fmt.Printf("Seed           : %d\n", *seedFlag)
	fmt.Printf("Elapsed        : %s\n", elapsed.Round(time.Millisecond))
}
