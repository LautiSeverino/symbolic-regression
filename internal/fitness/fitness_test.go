package fitness_test

import (
	"math"
	"testing"

	"github.com/LautiSeverino/symbolic-regression/internal/dataset"
	"github.com/LautiSeverino/symbolic-regression/internal/expression"
	"github.com/LautiSeverino/symbolic-regression/internal/fitness"
)

// ─── Helpers ──────────────────────────────────────────────────────────────

// linear2x1 construye (2 * x) + 1 — expresión canónica del spec.
func linear2x1() *expression.Node {
	return expression.NewBinary(expression.NodeAdd,
		expression.NewBinary(expression.NodeMul,
			expression.NewConstant(2),
			expression.NewVariable(),
		),
		expression.NewConstant(1),
	)
}

// canonicalDS es el dataset canónico del spec: {(1,3),(2,5),(3,7)}.
func canonicalDS() *dataset.Dataset {
	return dataset.New([]dataset.Point{
		{X: 1, Y: 3},
		{X: 2, Y: 5},
		{X: 3, Y: 7},
	})
}

// makeDS crea un dataset determinista de n puntos con f(x) = 2x+1.
func makeDS(n int) *dataset.Dataset {
	pts := make([]dataset.Point, n)
	for i := range pts {
		x := float64(i + 1)
		pts[i] = dataset.Point{X: x, Y: 2*x + 1}
	}
	return dataset.New(pts)
}

// ─── MSE exacto ───────────────────────────────────────────────────────────

func TestEvaluate_MSEZero(t *testing.T) {
	// f(x) = 2*x+1 evaluada sobre su propio dataset → MSE = 0 exacto.
	result, err := fitness.Evaluate(linear2x1(), canonicalDS(), 0)
	if err != nil {
		t.Fatalf("error inesperado: %v", err)
	}
	if result.MSE != 0 {
		t.Errorf("MSE = %v, want 0", result.MSE)
	}
	if result.Fitness != 0 {
		t.Errorf("Fitness = %v, want 0 (lambda=0, MSE=0)", result.Fitness)
	}
	if result.InvalidCount != 0 {
		t.Errorf("InvalidCount = %d, want 0", result.InvalidCount)
	}
}

func TestEvaluate_MSEKnown(t *testing.T) {
	// Dataset: y = x+1. Expresión: x (deliberadamente incorrecta, error = 1 en cada punto).
	// Cálculo manual: (1^2 + 1^2 + 1^2) / 3 = 1.0 — exacto en float64.
	ds := dataset.New([]dataset.Point{
		{X: 1, Y: 2},
		{X: 2, Y: 3},
		{X: 3, Y: 4},
	})
	result, err := fitness.Evaluate(expression.NewVariable(), ds, 0)
	if err != nil {
		t.Fatalf("error inesperado: %v", err)
	}
	if result.MSE != 1.0 {
		t.Errorf("MSE = %v, want 1.0", result.MSE)
	}
	if result.InvalidCount != 0 {
		t.Errorf("InvalidCount = %d, want 0", result.InvalidCount)
	}
}

// ─── Complejidad ───────────────────────────────────────────────────────────

func TestEvaluate_Complexity(t *testing.T) {
	ds := canonicalDS()

	// x: 1 nodo
	r1, err := fitness.Evaluate(expression.NewVariable(), ds, 0)
	if err != nil {
		t.Fatalf("variable: %v", err)
	}
	if r1.Complexity != 1 {
		t.Errorf("variable: Complexity = %d, want 1", r1.Complexity)
	}

	// x + 1: NodeAdd + NodeVariable + NodeConstant = 3 nodos
	xPlus1 := expression.NewBinary(expression.NodeAdd,
		expression.NewVariable(), expression.NewConstant(1))
	r2, err := fitness.Evaluate(xPlus1, ds, 0)
	if err != nil {
		t.Fatalf("x+1: %v", err)
	}
	if r2.Complexity != 3 {
		t.Errorf("x+1: Complexity = %d, want 3", r2.Complexity)
	}

	// (2*x)+1: NodeAdd + NodeMul + NodeConstant(2) + NodeVariable + NodeConstant(1) = 5
	r3, err := fitness.Evaluate(linear2x1(), ds, 0)
	if err != nil {
		t.Fatalf("2x+1: %v", err)
	}
	if r3.Complexity != 5 {
		t.Errorf("2x+1: Complexity = %d, want 5", r3.Complexity)
	}

	// Orden estrictamente creciente
	if !(r1.Complexity < r2.Complexity && r2.Complexity < r3.Complexity) {
		t.Errorf("se esperaba Complexity(x) < Complexity(x+1) < Complexity(2x+1): got %d, %d, %d",
			r1.Complexity, r2.Complexity, r3.Complexity)
	}
}

// ─── Fitness combinado ─────────────────────────────────────────────────────

func TestEvaluate_FitnessCombined(t *testing.T) {
	// Dataset y=x+1, expression x → MSE=1.0 exacto, Complexity=1.
	// Fitness = MSE + lambda * Complexity = 1.0 + 0.5*1 = 1.5 exacto.
	ds := dataset.New([]dataset.Point{
		{X: 1, Y: 2},
		{X: 2, Y: 3},
		{X: 3, Y: 4},
	})
	lambda := 0.5
	result, err := fitness.Evaluate(expression.NewVariable(), ds, lambda)
	if err != nil {
		t.Fatalf("error inesperado: %v", err)
	}
	wantFitness := result.MSE + lambda*float64(result.Complexity) // 1.0 + 0.5*1 = 1.5
	if result.Fitness != wantFitness {
		t.Errorf("Fitness = %v, want %v", result.Fitness, wantFitness)
	}
	if result.Fitness != 1.5 {
		t.Errorf("Fitness = %v, want 1.5", result.Fitness)
	}
}

func TestEvaluate_LambdaZero(t *testing.T) {
	// lambda=0: la penalización de complejidad no aporta → Fitness == MSE.
	result, err := fitness.Evaluate(expression.NewVariable(), canonicalDS(), 0)
	if err != nil {
		t.Fatalf("error inesperado: %v", err)
	}
	if result.Fitness != result.MSE {
		t.Errorf("lambda=0: Fitness=%v, MSE=%v, deben ser iguales", result.Fitness, result.MSE)
	}
}

func TestEvaluate_ComplexityPenaltyEffect(t *testing.T) {
	// Dos expresiones con MSE=0 pero distinta complejidad.
	// Con lambda>0, la más simple debe ganar en fitness.
	ds := canonicalDS()
	lambda := 0.01

	// simple: (2*x)+1 → complexity=5
	rSimple, err := fitness.Evaluate(linear2x1(), ds, lambda)
	if err != nil {
		t.Fatalf("simple: %v", err)
	}

	// complex: ((2*x)+1)+0 → complexity=7 (NodeAdd extra + NodeConstant(0))
	complexExpr := expression.NewBinary(expression.NodeAdd,
		linear2x1(), expression.NewConstant(0))
	rComplex, err := fitness.Evaluate(complexExpr, ds, lambda)
	if err != nil {
		t.Fatalf("complex: %v", err)
	}

	if rSimple.MSE != 0 || rComplex.MSE != 0 {
		t.Errorf("ambas deben tener MSE=0: simple=%v, complex=%v", rSimple.MSE, rComplex.MSE)
	}
	if rSimple.Complexity >= rComplex.Complexity {
		t.Errorf("simple.Complexity(%d) debe ser < complex.Complexity(%d)",
			rSimple.Complexity, rComplex.Complexity)
	}
	if rSimple.Fitness >= rComplex.Fitness {
		t.Errorf("simple.Fitness(%v) debe ser < complex.Fitness(%v)",
			rSimple.Fitness, rComplex.Fitness)
	}
}

// ─── EvalError ─────────────────────────────────────────────────────────────

func TestEvaluate_AllInvalid(t *testing.T) {
	// 1/0: todos los puntos producen división por cero.
	// InvalidCount = n; MSE = PenaltyPerInvalidPoint.
	divByZero := expression.NewBinary(expression.NodeDiv,
		expression.NewConstant(1), expression.NewConstant(0))
	ds := dataset.New([]dataset.Point{{X: 1, Y: 3}, {X: 2, Y: 5}})

	result, err := fitness.Evaluate(divByZero, ds, 0)
	if err != nil {
		t.Fatalf("error inesperado: %v", err)
	}
	if result.InvalidCount != 2 {
		t.Errorf("InvalidCount = %d, want 2", result.InvalidCount)
	}
	// MSE = (2 * PenaltyPerInvalidPoint) / 2 = PenaltyPerInvalidPoint
	if result.MSE != fitness.PenaltyPerInvalidPoint {
		t.Errorf("MSE = %v, want %v", result.MSE, fitness.PenaltyPerInvalidPoint)
	}
	// El resultado debe ser finito a pesar de la penalización
	if math.IsInf(result.Fitness, 0) || math.IsNaN(result.Fitness) {
		t.Errorf("Fitness debe ser finito, got %v", result.Fitness)
	}
}

func TestEvaluate_PartiallyInvalid(t *testing.T) {
	// x / (x - 1): falla en x=1, válido en x=0 y x=2.
	// Dataset: {(0,1),(1,2),(2,3)}
	//   x=0: pred=0/(−1)=0, sq=(1−0)²=1
	//   x=1: EvalError (denominador=0) → PenaltyPerInvalidPoint
	//   x=2: pred=2/1=2, sq=(3−2)²=1
	// MSE = (1 + PenaltyPerInvalidPoint + 1) / 3 = (PenaltyPerInvalidPoint + 2) / 3
	xDivXm1 := expression.NewBinary(expression.NodeDiv,
		expression.NewVariable(),
		expression.NewBinary(expression.NodeSub,
			expression.NewVariable(),
			expression.NewConstant(1),
		),
	)
	ds := dataset.New([]dataset.Point{{X: 0, Y: 1}, {X: 1, Y: 2}, {X: 2, Y: 3}})

	result, err := fitness.Evaluate(xDivXm1, ds, 0)
	if err != nil {
		t.Fatalf("error inesperado: %v", err)
	}
	if result.InvalidCount != 1 {
		t.Errorf("InvalidCount = %d, want 1", result.InvalidCount)
	}
	// (1e9 + 2) / 3 = 333333334.0 exacto en float64
	wantMSE := (fitness.PenaltyPerInvalidPoint + 2) / 3.0
	if result.MSE != wantMSE {
		t.Errorf("MSE = %v, want %v", result.MSE, wantMSE)
	}
}

func TestEvaluate_NoPanic(t *testing.T) {
	// Ninguna expresión inválida debe producir panic.
	ds := dataset.New([]dataset.Point{{X: 1, Y: 3}, {X: 2, Y: 5}})
	cases := []struct {
		name string
		expr *expression.Node
	}{
		{"div por cero constante",
			expression.NewBinary(expression.NodeDiv,
				expression.NewConstant(1), expression.NewConstant(0))},
		{"sqrt de negativo",
			expression.NewUnary(expression.NodeSqrt, expression.NewConstant(-1))},
		{"ln de cero",
			expression.NewUnary(expression.NodeLn, expression.NewConstant(0))},
		{"nodo nil", nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("panic inesperado: %v", r)
				}
			}()
			_, _ = fitness.Evaluate(tc.expr, ds, 0.01)
		})
	}
}

// ─── Dataset inválido ──────────────────────────────────────────────────────

func TestEvaluate_EmptyDataset(t *testing.T) {
	_, err := fitness.Evaluate(expression.NewVariable(), dataset.New(nil), 0)
	if err == nil {
		t.Error("se esperaba error para dataset vacío, se obtuvo nil")
	}
}

func TestEvaluate_NilDataset(t *testing.T) {
	_, err := fitness.Evaluate(expression.NewVariable(), nil, 0)
	if err == nil {
		t.Error("se esperaba error para dataset nil, se obtuvo nil")
	}
}

// ─── Seguridad numérica ────────────────────────────────────────────────────

func TestEvaluate_AllInvalidFiniteResult(t *testing.T) {
	// Con todos los puntos inválidos el MSE debe seguir siendo finito.
	divByZero := expression.NewBinary(expression.NodeDiv,
		expression.NewConstant(1), expression.NewConstant(0))
	result, err := fitness.Evaluate(divByZero, makeDS(10), 0)
	if err != nil {
		t.Fatalf("error inesperado: %v", err)
	}
	if math.IsInf(result.MSE, 0) || math.IsNaN(result.MSE) {
		t.Errorf("MSE debe ser finito con todos inválidos, got %v", result.MSE)
	}
	if math.IsInf(result.Fitness, 0) || math.IsNaN(result.Fitness) {
		t.Errorf("Fitness debe ser finito con todos inválidos, got %v", result.Fitness)
	}
}

func TestEvaluate_LambdaEdgeCases(t *testing.T) {
	// Verificar que lambda válido produce resultado finito y lambda no-finito devuelve error.
	ds := canonicalDS()
	expr := linear2x1() // MSE=0

	cases := []struct {
		name   string
		lambda float64
		wantOK bool
	}{
		{"cero", 0, true},
		{"muy pequeño", 1e-10, true},
		{"uno", 1.0, true},
		{"muy grande", 1e10, true},
		{"+Inf", math.Inf(1), false},
		{"-Inf", math.Inf(-1), false},
		{"NaN", math.NaN(), false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := fitness.Evaluate(expr, ds, tc.lambda)
			if tc.wantOK {
				if err != nil {
					t.Errorf("lambda=%v: error inesperado: %v", tc.lambda, err)
					return
				}
				if math.IsInf(result.Fitness, 0) || math.IsNaN(result.Fitness) {
					t.Errorf("lambda=%v: Fitness no-finito: %v", tc.lambda, result.Fitness)
				}
			} else {
				if err == nil {
					t.Errorf("lambda=%v: se esperaba error por resultado no-finito", tc.lambda)
				}
			}
		})
	}
}
