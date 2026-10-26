package genetic_test

// integration_test.go — Fase 12: pipeline completo con validación integrada.
//
// Diferencias respecto a Fase 11:
//   - NewEvolution recibe train y val; Run() asigna best.ValidationResult.
//   - Se elimina la llamada manual a fitness.Evaluate para validación.
//   - Se elimina el import de fitness (ya no es necesario aquí).

import (
	"math"
	"testing"

	"github.com/LautiSeverino/symbolic-regression/internal/dataset"
	"github.com/LautiSeverino/symbolic-regression/internal/expression"
	"github.com/LautiSeverino/symbolic-regression/internal/generator"
	"github.com/LautiSeverino/symbolic-regression/internal/genetic"
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
		Workers:        1,
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
//  3. genetic.NewEvolution + Run ejecuta la evolución sobre train y asigna
//     best.ValidationResult (lambda=0) sobre val al finalizar
//  4. Se comprueban training MSE y validation MSE directamente en best
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
	evol, err := genetic.NewEvolution(cfg, train, val)
	if err != nil {
		t.Fatalf("NewEvolution: %v", err)
	}
	best, err := evol.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// ValidationResult debe estar presente cuando se proporciona val
	if best.ValidationResult == nil {
		t.Fatal("ValidationResult es nil: Run() debe evaluar sobre val cuando se proporciona")
	}

	t.Logf("Expression     : %s", expression.Format(best.Expr))
	t.Logf("Training MSE   : %f", best.Result.MSE)
	t.Logf("Validation MSE : %f", best.ValidationResult.MSE)
	t.Logf("Complexity     : %d nodes", best.Result.Complexity)

	// Criterios de aceptación:
	// MSE < 1.0 es conservador; el demo real alcanza ~0.0001 con pop=500 gen=200.
	// El umbral evita que el test sea frágil ante variaciones menores en el RNG.
	const mseThreshold = 1.0

	if best.Result.MSE > mseThreshold {
		t.Errorf("Training MSE = %f, quería ≤ %f para f(x) = 2*x + 1",
			best.Result.MSE, mseThreshold)
	}
	if best.ValidationResult.MSE > mseThreshold {
		t.Errorf("Validation MSE = %f, quería ≤ %f (posible overfitting al training set)",
			best.ValidationResult.MSE, mseThreshold)
	}

	// La expresión no debe producir resultados inválidos en ningún punto del dataset completo.
	if best.Result.InvalidCount != 0 {
		t.Errorf("Training: %d puntos inválidos (división por cero / dominio inválido)",
			best.Result.InvalidCount)
	}
	if best.ValidationResult.InvalidCount != 0 {
		t.Errorf("Validation: %d puntos inválidos", best.ValidationResult.InvalidCount)
	}
}

// TestIntegration_LinearReproducible verifica que la misma seed produce
// exactamente el mismo resultado en el pipeline completo, incluyendo el split.
func TestIntegration_LinearReproducible(t *testing.T) {
	run := func() (string, float64) {
		ds := dataset.Generate(func(x float64) float64 { return 2*x + 1 }, 1, 20, 20)
		train, val, err := dataset.Split(ds, 0.80, 42)
		if err != nil {
			t.Fatalf("Split: %v", err)
		}
		cfg := integrationCfg(42)
		evol, err := genetic.NewEvolution(cfg, train, val)
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

// TestIntegration_LinearValidationFinite asegura que ValidationResult.MSE
// es siempre un valor finito, nunca NaN ni ±Inf.
func TestIntegration_LinearValidationFinite(t *testing.T) {
	ds := dataset.Generate(func(x float64) float64 { return 2*x + 1 }, 1, 20, 20)
	train, val, err := dataset.Split(ds, 0.80, 42)
	if err != nil {
		t.Fatalf("Split: %v", err)
	}

	cfg := integrationCfg(42)
	evol, err := genetic.NewEvolution(cfg, train, val)
	if err != nil {
		t.Fatalf("NewEvolution: %v", err)
	}
	best, err := evol.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if best.ValidationResult == nil {
		t.Fatal("ValidationResult es nil")
	}
	if math.IsNaN(best.ValidationResult.MSE) || math.IsInf(best.ValidationResult.MSE, 0) {
		t.Errorf("Validation MSE no es finito: %v", best.ValidationResult.MSE)
	}
}
