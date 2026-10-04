package genetic_test

// integration_test.go — Fase 7: tests de integración del pipeline completo.
//
// Diferencias con genetic_test.go (unit tests):
//   - Usan dataset.Generate + dataset.Split (pipeline completo, no solo NewEvolution)
//   - Evalúan validación por separado (fitness.Evaluate sobre val set)
//   - Comprueban MSE real, no solo que "mejore" o "no panic"
//
// No usan LoadCSV para evitar dependencia de rutas relativas de disco en CI.
// El CSV se testea independientemente en internal/dataset/dataset_test.go.

import (
	"math"
	"testing"

	"github.com/LautiSeverino/symbolic-regression/internal/dataset"
	"github.com/LautiSeverino/symbolic-regression/internal/expression"
	"github.com/LautiSeverino/symbolic-regression/internal/fitness"
	"github.com/LautiSeverino/symbolic-regression/internal/genetic"
	"github.com/LautiSeverino/symbolic-regression/internal/generator"
)

// integrationCfg devuelve una configuración equilibrada para tests de integración.
// Parámetros intencionalmente menores que el demo de main.go para mantener
// el tiempo de test bajo (< 500ms en hardware modesto).
func integrationCfg(seed int64) genetic.Config {
	return genetic.Config{
		PopulationSize: 300,
		Generations:    150,
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
			AllowedUnaryOps:     nil,
			ConstantMin:         -5,
			ConstantMax:         5,
			VariableProbability: 0.50,
			TerminalProbability: 0.30,
			Seed:                seed + 1,
		},
	}
}

// TestIntegration_LinearPipeline verifica el pipeline completo sobre f(x) = 2*x + 1.
//
// Pasos cubiertos:
//  1. dataset.Generate crea el dataset sintético
//  2. dataset.Split divide en train / validation
//  3. genetic.NewEvolution + Run ejecuta la evolución sobre train
//  4. fitness.Evaluate mide el error final sobre val
//
// Criterio de éxito: el GA debe encontrar una expresión con MSE
// razonablemente bajo tanto en training como en validation.
// No se exige MSE=0 porque la penalización de complejidad puede preferir
// una expresión ligeramente imprecisa pero más simple (comportamiento correcto).
func TestIntegration_LinearPipeline(t *testing.T) {
	// 1. Generar dataset: f(x) = 2*x + 1, x ∈ [1, 20], 20 puntos
	ds := dataset.Generate(func(x float64) float64 { return 2*x + 1 }, 1, 20, 20)
	if ds.Len() != 20 {
		t.Fatalf("dataset: esperaba 20 puntos, got %d", ds.Len())
	}

	// 2. Split 80% train / 20% validation
	train, val, err := dataset.Split(ds, 0.80, 42)
	if err != nil {
		t.Fatalf("Split: %v", err)
	}
	if train.Len()+val.Len() != ds.Len() {
		t.Fatalf("Split: train(%d)+val(%d) ≠ total(%d)", train.Len(), val.Len(), ds.Len())
	}

	// 3. Ejecutar evolución
	cfg := integrationCfg(42)
	evol, err := genetic.NewEvolution(cfg, train)
	if err != nil {
		t.Fatalf("NewEvolution: %v", err)
	}
	best, err := evol.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// 4. Evaluar en validation (lambda=0: MSE puro, sin penalización de complejidad)
	valResult, err := fitness.Evaluate(best.Expr, val, 0)
	if err != nil {
		t.Fatalf("Evaluate validation: %v", err)
	}

	t.Logf("Expression     : %s", expression.Format(best.Expr))
	t.Logf("Training MSE   : %f", best.Result.MSE)
	t.Logf("Validation MSE : %f", valResult.MSE)
	t.Logf("Complexity     : %d nodes", best.Result.Complexity)

	// Criterios de aceptación:
	// MSE < 1.0 es conservador; el demo real alcanza ~0.0001 con pop=500 gen=200.
	// El umbral evita que el test sea frágil ante variaciones menores en el RNG.
	const mseThreshold = 1.0

	if best.Result.MSE > mseThreshold {
		t.Errorf("Training MSE = %f, quería ≤ %f para f(x) = 2*x + 1",
			best.Result.MSE, mseThreshold)
	}
	if valResult.MSE > mseThreshold {
		t.Errorf("Validation MSE = %f, quería ≤ %f (posible overfitting al training set)",
			valResult.MSE, mseThreshold)
	}

	// La expresión no debe producir resultados inválidos en ningún punto del dataset completo.
	if best.Result.InvalidCount != 0 {
		t.Errorf("Training: %d puntos inválidos (división por cero / dominio inválido)",
			best.Result.InvalidCount)
	}
	if valResult.InvalidCount != 0 {
		t.Errorf("Validation: %d puntos inválidos", valResult.InvalidCount)
	}
}

// TestIntegration_LinearReproducible verifica que la misma seed produce
// exactamente el mismo resultado en el pipeline completo, incluyendo el split.
func TestIntegration_LinearReproducible(t *testing.T) {
	run := func() (string, float64) {
		ds := dataset.Generate(func(x float64) float64 { return 2*x + 1 }, 1, 20, 20)
		train, _, err := dataset.Split(ds, 0.80, 42)
		if err != nil {
			t.Fatalf("Split: %v", err)
		}
		cfg := integrationCfg(42)
		evol, err := genetic.NewEvolution(cfg, train)
		if err != nil {
			t.Fatalf("NewEvolution: %v", err)
		}
		best, err := evol.Run()
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		return expression.Format(best.Expr), best.Result.Fitness
	}

	expr1, fit1 := run()
	expr2, fit2 := run()

	if expr1 != expr2 {
		t.Errorf("misma seed → expresiones distintas:\n  %s\n  %s", expr1, expr2)
	}
	if fit1 != fit2 {
		t.Errorf("misma seed → fitness distintos: %v vs %v", fit1, fit2)
	}
}

// TestIntegration_LinearValidationFinite asegura que los resultados de validation
// son siempre valores finitos, nunca NaN ni ±Inf.
func TestIntegration_LinearValidationFinite(t *testing.T) {
	ds := dataset.Generate(func(x float64) float64 { return 2*x + 1 }, 1, 20, 20)
	train, val, err := dataset.Split(ds, 0.80, 42)
	if err != nil {
		t.Fatalf("Split: %v", err)
	}

	cfg := integrationCfg(42)
	evol, err := genetic.NewEvolution(cfg, train)
	if err != nil {
		t.Fatalf("NewEvolution: %v", err)
	}
	best, err := evol.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	valResult, err := fitness.Evaluate(best.Expr, val, 0)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}

	if math.IsNaN(valResult.MSE) || math.IsInf(valResult.MSE, 0) {
		t.Errorf("Validation MSE no es finito: %v", valResult.MSE)
	}
}
