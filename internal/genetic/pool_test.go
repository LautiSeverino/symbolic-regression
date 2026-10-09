package genetic_test

// pool_test.go — Fase 10: tests del worker pool y paralelización.
//
// Cubre:
//  1. Configuración de Workers (valores válidos e inválidos)
//  2. Workers=1 produce resultados correctos
//  3. Workers>1 produce resultados correctos
//  4. Determinismo: Workers=1 vs Workers>1 con misma seed → mismo resultado
//  5. Población pequeña (Workers > PopulationSize)
//  6. Expresiones con evaluación inválida no producen panic
//  7. InitPopulation con múltiples workers
//  8. NextGeneration con múltiples workers
//  9. Benchmarks informales de Workers=1/2/4/N

import (
	"fmt"
	"runtime"
	"testing"
	"time"

	"github.com/LautiSeverino/symbolic-regression/internal/dataset"
	"github.com/LautiSeverino/symbolic-regression/internal/expression"
	"github.com/LautiSeverino/symbolic-regression/internal/genetic"
	"github.com/LautiSeverino/symbolic-regression/internal/generator"
)

// ─── Helpers ──────────────────────────────────────────────────────────────

// poolCfg devuelve una config válida con el número de workers dado.
func poolCfg(workers int) genetic.Config {
	return genetic.Config{
		PopulationSize: 40,
		Generations:    10,
		TournamentSize: 3,
		ElitismRate:    0.05,
		MutationRate:   0.10,
		CrossoverRate:  0.80,
		Lambda:         0.01,
		Seed:           42,
		Workers:        workers,
		GenConfig: generator.Config{
			MaxDepth:            4,
			MaxNodes:            15,
			AllowedBinaryOps:    []expression.NodeType{expression.NodeAdd, expression.NodeMul},
			AllowedUnaryOps:     nil,
			ConstantMin:         -5,
			ConstantMax:         5,
			VariableProbability: 0.5,
			TerminalProbability: 0.3,
			Seed:                42,
		},
	}
}

// poolDS devuelve el dataset canónico y = 2*x + 1.
func poolDS() *dataset.Dataset {
	return dataset.New([]dataset.Point{
		{X: 1, Y: 3}, {X: 2, Y: 5}, {X: 3, Y: 7},
		{X: 4, Y: 9}, {X: 5, Y: 11},
	})
}

// runEvolution ejecuta la evolución completa con la configuración dada.
func runEvolution(t *testing.T, cfg genetic.Config) *genetic.Individual {
	t.Helper()
	evol, err := genetic.NewEvolution(cfg, poolDS())
	if err != nil {
		t.Fatalf("NewEvolution(workers=%d): %v", cfg.Workers, err)
	}
	best, err := evol.Run()
	if err != nil {
		t.Fatalf("Run(workers=%d): %v", cfg.Workers, err)
	}
	return best
}

// ─── 1. Configuración de Workers ─────────────────────────────────────────

func TestWorkers_Zero_NormalizedToOne(t *testing.T) {
	// Workers=0 debe ser normalizado a 1 sin error.
	cfg := poolCfg(0)
	_, err := genetic.NewEvolution(cfg, poolDS())
	if err != nil {
		t.Errorf("Workers=0: error inesperado (debe normalizarse a 1): %v", err)
	}
}

func TestWorkers_Negative_NormalizedToOne(t *testing.T) {
	// Workers negativos deben normalizarse a 1 sin error.
	for _, w := range []int{-1, -10, -100} {
		cfg := poolCfg(w)
		_, err := genetic.NewEvolution(cfg, poolDS())
		if err != nil {
			t.Errorf("Workers=%d: error inesperado (debe normalizarse a 1): %v", w, err)
		}
	}
}

func TestWorkers_ValidValues(t *testing.T) {
	// Workers >= 1 deben ser aceptados.
	for _, w := range []int{1, 2, 4, 8, 16} {
		cfg := poolCfg(w)
		evol, err := genetic.NewEvolution(cfg, poolDS())
		if err != nil {
			t.Errorf("Workers=%d: error inesperado: %v", w, err)
		}
		if evol == nil {
			t.Errorf("Workers=%d: NewEvolution devolvió nil", w)
		}
	}
}

// ─── 2. Workers=1 produce resultados correctos ────────────────────────────

func TestWorkers_One_CorrectResults(t *testing.T) {
	best := runEvolution(t, poolCfg(1))

	if best.Expr == nil {
		t.Fatal("Workers=1: Expr es nil")
	}
	if best.Result.Fitness < 0 {
		t.Errorf("Workers=1: Fitness negativo: %v", best.Result.Fitness)
	}
	if best.Result.Complexity < 1 {
		t.Errorf("Workers=1: Complexity < 1: %d", best.Result.Complexity)
	}
}

// ─── 3. Workers>1 produce resultados correctos ────────────────────────────

func TestWorkers_Multiple_CorrectResults(t *testing.T) {
	for _, w := range []int{2, 4, 8} {
		w := w
		t.Run(fmt.Sprintf("workers=%d", w), func(t *testing.T) {
			best := runEvolution(t, poolCfg(w))
			if best.Expr == nil {
				t.Fatalf("Workers=%d: Expr es nil", w)
			}
			if best.Result.Fitness < 0 {
				t.Errorf("Workers=%d: Fitness negativo: %v", w, best.Result.Fitness)
			}
			if best.Result.Complexity < 1 {
				t.Errorf("Workers=%d: Complexity < 1: %d", w, best.Result.Complexity)
			}
		})
	}
}

// ─── 4. Determinismo: Workers=1 vs Workers>1 ─────────────────────────────

// TestWorkers_Determinism_OneVsMany es el test más importante de esta fase.
//
// Garantía: misma seed + misma config → mismo resultado sin importar Workers.
//
// Por qué funciona:
//   - La generación de expresiones (e.gen, e.rng) ocurre secuencialmente y
//     no es afectada por el número de workers.
//   - La evaluación de fitness es una función pura: misma expresión + mismo
//     dataset → mismo resultado siempre.
//   - Los resultados se ensamblan por índice, no por orden de llegada.
//   - Por lo tanto, la población tras cada generación es idéntica.
func TestWorkers_Determinism_OneVsMany(t *testing.T) {
	cases := []struct {
		a, b int
	}{
		{1, 2},
		{1, 4},
		{1, 8},
		{2, 4},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(fmt.Sprintf("workers_%d_vs_%d", tc.a, tc.b), func(t *testing.T) {
			bestA := runEvolution(t, poolCfg(tc.a))
			bestB := runEvolution(t, poolCfg(tc.b))

			exprA := expression.Format(bestA.Expr)
			exprB := expression.Format(bestB.Expr)
			if exprA != exprB {
				t.Errorf("Workers=%d vs Workers=%d: misma seed produce expresiones distintas:\n  %s\n  %s",
					tc.a, tc.b, exprA, exprB)
			}
			if bestA.Result.Fitness != bestB.Result.Fitness {
				t.Errorf("Workers=%d vs Workers=%d: fitness distintos: %v vs %v",
					tc.a, tc.b, bestA.Result.Fitness, bestB.Result.Fitness)
			}
		})
	}
}

// ─── 5. Población pequeña (Workers > PopulationSize) ─────────────────────

func TestWorkers_MoreWorkersThanPopulation(t *testing.T) {
	// Si Workers > PopulationSize, el pool debe ajustar internamente
	// y no producir goroutines ociosas ni deadlocks.
	cfg := poolCfg(100) // 100 workers, población de 40
	cfg.PopulationSize = 5
	cfg.TournamentSize = 2
	cfg.Generations = 3

	best := runEvolution(t, cfg)
	if best == nil {
		t.Fatal("resultado nil con Workers > PopulationSize")
	}
	if best.Expr == nil {
		t.Fatal("Expr nil con Workers > PopulationSize")
	}
}

// ─── 6. Expresiones inválidas — sin panic ────────────────────────────────

func TestWorkers_InvalidExpressions_NoPanic(t *testing.T) {
	// Dataset que incluye x=0 para activar división por cero en algunas expresiones.
	ds := dataset.New([]dataset.Point{
		{X: -2, Y: 1}, {X: -1, Y: 1}, {X: 0, Y: 1},
		{X: 1, Y: 1}, {X: 2, Y: 1},
	})

	cfg := poolCfg(4)
	cfg.GenConfig.AllowedBinaryOps = []expression.NodeType{
		expression.NodeAdd, expression.NodeMul, expression.NodeDiv,
	}

	defer func() {
		if r := recover(); r != nil {
			t.Errorf("panic inesperado con expresiones inválidas: %v", r)
		}
	}()

	evol, err := genetic.NewEvolution(cfg, ds)
	if err != nil {
		t.Fatalf("NewEvolution: %v", err)
	}
	_, err = evol.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
}

// ─── 7. InitPopulation con múltiples workers ─────────────────────────────

func TestWorkers_InitPopulation_MultipleWorkers(t *testing.T) {
	// InitPopulation debe producir exactamente PopulationSize individuos
	// con fitness finito, con cualquier número de workers.
	for _, w := range []int{1, 2, 4} {
		w := w
		t.Run(fmt.Sprintf("workers=%d", w), func(t *testing.T) {
			cfg := poolCfg(w)
			evol, _ := genetic.NewEvolution(cfg, poolDS())
			pop := evol.InitPopulation()

			if len(pop) != cfg.PopulationSize {
				t.Errorf("Workers=%d: len(pop)=%d, want %d", w, len(pop), cfg.PopulationSize)
			}
			for i, ind := range pop {
				if ind.Expr == nil {
					t.Errorf("Workers=%d individuo[%d]: Expr nil", w, i)
				}
				if ind.Result.Fitness != ind.Result.Fitness { // NaN check
					t.Errorf("Workers=%d individuo[%d]: Fitness NaN", w, i)
				}
			}
		})
	}
}

// ─── 8. NextGeneration con múltiples workers ─────────────────────────────

func TestWorkers_NextGeneration_PreservesSize(t *testing.T) {
	// NextGeneration debe producir exactamente PopulationSize individuos
	// con cualquier número de workers.
	for _, w := range []int{1, 2, 4} {
		w := w
		t.Run(fmt.Sprintf("workers=%d", w), func(t *testing.T) {
			cfg := poolCfg(w)
			cfg.Generations = 5
			evol, _ := genetic.NewEvolution(cfg, poolDS())
			pop := evol.InitPopulation()
			for range 5 {
				pop = evol.NextGeneration(pop)
				if len(pop) != cfg.PopulationSize {
					t.Errorf("Workers=%d: len(pop)=%d, want %d", w, len(pop), cfg.PopulationSize)
				}
			}
		})
	}
}

// ─── 9. Determinismo de InitPopulation ───────────────────────────────────

func TestWorkers_InitPopulation_Deterministic(t *testing.T) {
	// Misma seed con Workers=1 y Workers=4 debe producir la misma población inicial.
	makeInitPop := func(workers int) []string {
		cfg := poolCfg(workers)
		evol, _ := genetic.NewEvolution(cfg, poolDS())
		pop := evol.InitPopulation()
		formats := make([]string, len(pop))
		for i, ind := range pop {
			formats[i] = expression.Format(ind.Expr)
		}
		return formats
	}

	pop1 := makeInitPop(1)
	pop4 := makeInitPop(4)

	if len(pop1) != len(pop4) {
		t.Fatalf("tamaños distintos: %d vs %d", len(pop1), len(pop4))
	}
	for i := range pop1 {
		if pop1[i] != pop4[i] {
			t.Errorf("individuo[%d]: Workers=1 vs Workers=4 produjeron expresiones distintas:\n  %s\n  %s",
				i, pop1[i], pop4[i])
		}
	}
}

// ─── 10. GlobalBest nunca empeora con Workers>1 ──────────────────────────

func TestWorkers_GlobalBestNeverWorsens_Parallel(t *testing.T) {
	cfg := poolCfg(4)
	cfg.Generations = 15

	var history []float64
	cfg.OnGeneration = func(gen int, best genetic.Individual) {
		history = append(history, best.Result.Fitness)
	}

	evol, _ := genetic.NewEvolution(cfg, poolDS())
	_, err := evol.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	for i := 1; i < len(history); i++ {
		if history[i] > history[i-1]+1e-10 {
			t.Errorf("gen %d: globalBest empeoró de %v a %v (Workers=4)", i, history[i-1], history[i])
		}
	}
}

// ─── 11. NoPanic con Workers>1 ───────────────────────────────────────────

func TestWorkers_NoPanic_Parallel(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("panic con Workers=4: %v", r)
		}
	}()
	cfg := poolCfg(4)
	evol, _ := genetic.NewEvolution(cfg, poolDS())
	_, err := evol.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
}

// ─── 12. Benchmarks informales: Workers=1/2/4/N ──────────────────────────
//
// Correr con: go test -bench=BenchmarkWorkers -benchtime=3s ./internal/genetic/
// No forman parte de los benchmarks formales de Fase 16.
// Son una primera observación de la mejora de rendimiento.

func benchmarkWorkers(b *testing.B, workers int) {
	b.Helper()
	ds := dataset.New(func() []dataset.Point {
		pts := make([]dataset.Point, 50)
		for i := range pts {
			x := float64(i + 1)
			pts[i] = dataset.Point{X: x, Y: 2*x + 1}
		}
		return pts
	}())

	cfg := genetic.Config{
		PopulationSize: 500,
		Generations:    20,
		TournamentSize: 3,
		ElitismRate:    0.05,
		MutationRate:   0.10,
		CrossoverRate:  0.80,
		Lambda:         0.01,
		Seed:           42,
		Workers:        workers,
		GenConfig: generator.Config{
			MaxDepth:         5,
			MaxNodes:         30,
			AllowedBinaryOps: []expression.NodeType{expression.NodeAdd, expression.NodeSub, expression.NodeMul, expression.NodeDiv},
			AllowedUnaryOps:  nil,
			ConstantMin:      -5,
			ConstantMax:      5,
			VariableProbability: 0.5,
			TerminalProbability: 0.3,
			Seed:             42,
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evol, err := genetic.NewEvolution(cfg, ds)
		if err != nil {
			b.Fatalf("NewEvolution: %v", err)
		}
		if _, err := evol.Run(); err != nil {
			b.Fatalf("Run: %v", err)
		}
	}
}

func BenchmarkWorkers_1(b *testing.B)  { benchmarkWorkers(b, 1) }
func BenchmarkWorkers_2(b *testing.B)  { benchmarkWorkers(b, 2) }
func BenchmarkWorkers_4(b *testing.B)  { benchmarkWorkers(b, 4) }
func BenchmarkWorkers_N(b *testing.B)  { benchmarkWorkers(b, runtime.NumCPU()) }

// ─── 13. Medición simple inline (no un benchmark formal) ─────────────────

// TestWorkers_SimpleTiming imprime tiempos comparativos para Workers=1/2/4/N.
// No falla por timing (los resultados dependen del hardware).
// Skipeado en modo -short para no afectar CI.
func TestWorkers_SimpleTiming(t *testing.T) {
	if testing.Short() {
		t.Skip("timing test skipped in short mode")
	}

	ds := dataset.New(func() []dataset.Point {
		pts := make([]dataset.Point, 100)
		for i := range pts {
			x := float64(i + 1)
			pts[i] = dataset.Point{X: x, Y: 2*x + 1}
		}
		return pts
	}())

	workerCounts := []int{1, 2, 4, runtime.NumCPU()}
	times := make([]time.Duration, len(workerCounts))

	for i, w := range workerCounts {
		cfg := genetic.Config{
			PopulationSize: 500,
			Generations:    30,
			TournamentSize: 3,
			ElitismRate:    0.05,
			MutationRate:   0.10,
			CrossoverRate:  0.80,
			Lambda:         0.01,
			Seed:           42,
			Workers:        w,
			GenConfig: generator.Config{
				MaxDepth:            5,
				MaxNodes:            30,
				AllowedBinaryOps:    []expression.NodeType{expression.NodeAdd, expression.NodeSub, expression.NodeMul, expression.NodeDiv},
				AllowedUnaryOps:     nil,
				ConstantMin:         -5,
				ConstantMax:         5,
				VariableProbability: 0.5,
				TerminalProbability: 0.3,
				Seed:                42,
			},
		}
		evol, err := genetic.NewEvolution(cfg, ds)
		if err != nil {
			t.Fatalf("workers=%d: %v", w, err)
		}
		start := time.Now()
		if _, err := evol.Run(); err != nil {
			t.Fatalf("Run workers=%d: %v", w, err)
		}
		times[i] = time.Since(start)
	}

	t.Log("─── Medición informal de Workers ────────────────────")
	t.Logf("  %-10s  %12s  %8s", "Workers", "Elapsed", "Speedup")
	baseline := times[0]
	for i, w := range workerCounts {
		speedup := float64(baseline) / float64(times[i])
		t.Logf("  %-10d  %12s  %8.2fx", w, times[i].Round(time.Millisecond), speedup)
	}
	t.Log("────────────────────────────────────────────────────")
	t.Log("  Nota: overhead del canal domina con pools pequeños.")
	t.Log("  Los benchmarks formales son Fase 16.")
}
