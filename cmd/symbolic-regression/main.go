// Fase 7 — Demostración end-to-end: regresión simbólica sobre f(x) = 2*x + 1.
//
// El programa carga un dataset CSV, lo divide en training/validation,
// ejecuta el algoritmo genético de forma secuencial e imprime el resultado.
// No recibe la fórmula original; únicamente los datos (x, y).
//
// Uso:
//
//	go run ./cmd/symbolic-regression
//	go run ./cmd/symbolic-regression datasets/linear.csv
package main

import (
	"fmt"
	"os"
	"time"

	"github.com/LautiSeverino/symbolic-regression/internal/dataset"
	"github.com/LautiSeverino/symbolic-regression/internal/expression"
	"github.com/LautiSeverino/symbolic-regression/internal/fitness"
	"github.com/LautiSeverino/symbolic-regression/internal/genetic"
	"github.com/LautiSeverino/symbolic-regression/internal/generator"
)

// Parámetros del demo de Fase 7.
// La CLI completa (--data, --profile, --seed, etc.) se implementa en Fase 15.
const (
	defaultDataset = "datasets/linear.csv"
	trainRatio     = 0.80
	seed           = int64(42)
	printEvery     = 10 // frecuencia de impresión durante la evolución
)

func main() {
	// Permitir pasar la ruta del dataset como único argumento posicional.
	dataPath := defaultDataset
	if len(os.Args) > 1 {
		dataPath = os.Args[1]
	}

	fmt.Println("══════════════════════════════════════════")
	fmt.Println("  Regresión Simbólica Evolutiva — Fase 7")
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
	train, val, err := dataset.Split(ds, trainRatio, seed)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error al dividir dataset: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Split    : %.0f%% train = %d puntos | %.0f%% val = %d puntos\n",
		trainRatio*100, train.Len(), (1-trainRatio)*100, val.Len())
	fmt.Println()

	// ── 3. Configuración del algoritmo genético ───────────────────────────
	cfg := genetic.Config{
		PopulationSize: 500,
		Generations:    200,
		TournamentSize: 3,
		ElitismRate:    0.05,
		MutationRate:   0.10,
		CrossoverRate:  0.80,
		Lambda:         0.01,
		Seed:           seed,
		GenConfig: generator.Config{
			MaxDepth: 5,
			MaxNodes: 30,
			AllowedBinaryOps: []expression.NodeType{
				expression.NodeAdd, expression.NodeSub,
				expression.NodeMul, expression.NodeDiv,
			},
			AllowedUnaryOps:     nil, // sin funciones trascendentes (datos lineales)
			ConstantMin:         -5,
			ConstantMax:         5,
			VariableProbability: 0.50,
			TerminalProbability: 0.30,
			Seed:                seed + 1, // semilla independiente del GA
		},
	}

	fmt.Printf("Config   : pop=%d | gen=%d | depth=%d | nodes=%d | seed=%d\n",
		cfg.PopulationSize, cfg.Generations,
		cfg.GenConfig.MaxDepth, cfg.GenConfig.MaxNodes, seed)
	fmt.Printf("Fitness  : MSE + λ·complexity  (λ=%.3f)\n", cfg.Lambda)
	fmt.Println()
	fmt.Printf("%-12s  %-14s  %-14s  %-8s  %s\n",
		"Generation", "MSE", "Fitness", "Nodes", "Expression")
	fmt.Println("──────────────────────────────────────────────────────────────────────")

	// ── 4. Callback de progreso ───────────────────────────────────────────
	start := time.Now()
	cfg.OnGeneration = func(gen int, best genetic.Individual) {
		// Imprimir la gen inicial (0), cada printEvery generaciones y la final.
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
	// lambda=0 para el MSE puro sin penalización de complejidad.
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
	fmt.Printf("Workers        : 1 (secuencial — ver Fase 10)\n")
	fmt.Printf("Seed           : %d\n", seed)
	fmt.Printf("Elapsed        : %s\n", elapsed.Round(time.Millisecond))
}
