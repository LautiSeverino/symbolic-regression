// Regresión Simbólica Evolutiva — Fase 15: CLI completa.
//
// El programa carga un dataset CSV, lo divide en training/validation,
// ejecuta el algoritmo genético con el perfil elegido (y overrides opcionales)
// e imprime el resultado. No recibe la fórmula original; únicamente los datos (x, y).
//
// Uso:
//
//	go run ./cmd/symbolic-regression [--data DATASET] [--profile PERFIL] [--seed N] [opciones]
//	go run ./cmd/symbolic-regression [--profile PERFIL] [--seed N] [opciones] DATASET
//
// Ejemplos:
//
//	go run ./cmd/symbolic-regression --data datasets/quadratic.csv --profile balanced --seed 42
//	go run ./cmd/symbolic-regression --population 2000 --generations 500 --workers 8 --seed 42
//	go run ./cmd/symbolic-regression datasets/linear.csv
package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/LautiSeverino/symbolic-regression/internal/config"
	"github.com/LautiSeverino/symbolic-regression/internal/dataset"
	"github.com/LautiSeverino/symbolic-regression/internal/expression"
	"github.com/LautiSeverino/symbolic-regression/internal/genetic"
)

const (
	trainRatio = 0.80
	printEvery = 10
)

func main() {
	opts, err := parseCLI(os.Args[1:], os.Stderr)
	if err != nil {
		if errors.Is(err, flag.ErrHelp) {
			os.Exit(0)
		}
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	// ── 1. Cargar perfil ───────────────────────────────────────────────────
	cfg, err := config.Parse(opts.Profile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	// Advertencia específica para BENCHMARK sin seed explícita.
	if strings.EqualFold(strings.TrimSpace(opts.Profile), "benchmark") && !opts.Explicit["seed"] {
		fmt.Fprintln(os.Stderr,
			"aviso: perfil benchmark usado sin --seed explícito; "+
				"los resultados pueden no ser comparables entre ejecuciones")
	}

	// ── 2. Asignar seeds (siempre; independientes del perfil) ──────────────
	cfg.Seed = opts.Seed
	cfg.GenConfig.Seed = opts.Seed + 1

	// ── 3. Aplicar overrides explícitos ────────────────────────────────────
	cfg = applyConfigOverrides(cfg, opts)
	cfg = normalizeConfig(cfg)

	// ── 4. Cargar dataset ──────────────────────────────────────────────────
	ds, err := dataset.LoadCSV(opts.DataPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error al cargar %q: %v\n", opts.DataPath, err)
		os.Exit(1)
	}

	// ── 5. División train / validation ────────────────────────────────────
	train, val, err := dataset.Split(ds, trainRatio, opts.Seed)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error al dividir dataset: %v\n", err)
		os.Exit(1)
	}

	// ── 6. Información de configuración ───────────────────────────────────
	fmt.Println("══════════════════════════════════════════")
	fmt.Println("  Regresión Simbólica Evolutiva — Fase 15")
	fmt.Println("══════════════════════════════════════════")
	fmt.Println()
	fmt.Printf("Dataset  : %s  (%d puntos)\n", opts.DataPath, ds.Len())
	fmt.Printf("Split    : %.0f%% train = %d puntos | %.0f%% val = %d puntos\n",
		trainRatio*100, train.Len(), (1-trainRatio)*100, val.Len())
	fmt.Println()
	fmt.Printf("Profile  : %s\n", strings.ToLower(opts.Profile))
	fmt.Printf("Config   : pop=%d | gen=%d | depth=%d | nodes=%d | workers=%d | seed=%d\n",
		cfg.PopulationSize, cfg.Generations,
		cfg.GenConfig.MaxDepth, cfg.GenConfig.MaxNodes, cfg.Workers, opts.Seed)
	fmt.Printf("Fitness  : MSE + λ·complexity  (λ=%.3f)\n", cfg.Lambda)
	fmt.Println()

	fmt.Printf("%-12s  %-14s  %-14s  %-8s  %s\n",
		"Generation", "MSE", "Fitness", "Nodes", "Expression")
	fmt.Println("──────────────────────────────────────────────────────────────────────")

	// ── 7. Callback de progreso ───────────────────────────────────────────
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

	// ── 8. Crear y ejecutar la evolución ──────────────────────────────────
	evol, err := genetic.NewEvolution(cfg, train, val)
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

	// ── 9. Resultado final ────────────────────────────────────────────────
	eliteCount := cfg.EliteCount()
	evaluated := cfg.PopulationSize + best.GenerationsRun*(cfg.PopulationSize-eliteCount)
	totalEvals := evaluated * train.Len()

	fmt.Println()
	fmt.Println("══════════════════════════════════════════")
	fmt.Println("  RESULT")
	fmt.Println("══════════════════════════════════════════")
	fmt.Println()
	fmt.Printf("Expression     : %s\n", expression.Format(best.Expr))
	fmt.Println()
	fmt.Printf("Training MSE   : %.10f\n", best.Result.MSE)
	if best.ValidationResult != nil {
		fmt.Printf("Validation MSE : %.10f\n", best.ValidationResult.MSE)
	}
	fmt.Println()
	fmt.Printf("Complexity     : %d nodes\n", best.Result.Complexity)
	fmt.Println()
	fmt.Printf("Generations    : %d\n", best.GenerationsRun)
	fmt.Printf("Population     : %d\n", cfg.PopulationSize)
	fmt.Printf("Total evals    : %d\n", totalEvals)
	fmt.Printf("Workers        : %d\n", cfg.Workers)
	fmt.Printf("Profile        : %s\n", strings.ToLower(opts.Profile))
	fmt.Printf("Seed           : %d\n", opts.Seed)
	fmt.Printf("Elapsed        : %s\n", elapsed.Round(time.Millisecond))
}
