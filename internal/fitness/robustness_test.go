package fitness_test

// robustness_test.go — Fase 9: robustez de fitness.Evaluate.
//
// Complementa fitness_test.go cubriendo casos no presentes allí:
//
//  1. Todos los operadores de Fase 8 producen MSE finito cuando son válidos
//  2. Todos los operadores de Fase 8 producen penalización cuando son inválidos
//  3. Overflow en el error cuadrático (pred ≈ ±MaxFloat64, Y de signo contrario)
//  4. Expresiones que producen resultados muy grandes pero finitos
//  5. Pipeline con constantes no finitas (nueva conducta Fase 9)
//  6. Invariantes: fitness ≥ 0, MSE ≥ 0, Complexity ≥ 1

import (
	"math"
	"testing"

	"github.com/LautiSeverino/symbolic-regression/internal/dataset"
	"github.com/LautiSeverino/symbolic-regression/internal/expression"
	"github.com/LautiSeverino/symbolic-regression/internal/fitness"
)

// ─── Helpers ──────────────────────────────────────────────────────────────

// positiveDS crea un dataset con x ∈ [1, n] e y = f(x).
func positiveDS(n int, f func(float64) float64) *dataset.Dataset {
	pts := make([]dataset.Point, n)
	for i := range pts {
		x := float64(i + 1)
		pts[i] = dataset.Point{X: x, Y: f(x)}
	}
	return dataset.New(pts)
}

// simpleDS devuelve el dataset canónico {(1,3),(2,5),(3,7)}.
func simpleDS() *dataset.Dataset {
	return dataset.New([]dataset.Point{
		{X: 1, Y: 3}, {X: 2, Y: 5}, {X: 3, Y: 7},
	})
}

// ══════════════════════════════════════════════════════════════════════════
// 1. Operadores de Fase 8 — MSE finito en expresiones válidas
// ══════════════════════════════════════════════════════════════════════════

func TestFitness_Phase8_ValidOps_FiniteMSE(t *testing.T) {
	// Dataset con x ∈ [1, 5], puntos positivos para que sqrt y ln sean siempre válidos.
	ds := positiveDS(5, func(x float64) float64 { return x })

	cases := []struct {
		name string
		expr *expression.Node
	}{
		// NodePow: x^2
		{"x^2", expression.NewBinary(expression.NodePow,
			expression.NewVariable(), expression.NewConstant(2))},
		// NodeAbs: abs(x)
		{"abs(x)", expression.NewUnary(expression.NodeAbs, expression.NewVariable())},
		// NodeExp: exp(x) — puede ser grande pero finito para x ∈ [1,5]
		{"exp(x)", expression.NewUnary(expression.NodeExp, expression.NewVariable())},
		// NodeSin: sin(x) — siempre en [-1,1]
		{"sin(x)", expression.NewUnary(expression.NodeSin, expression.NewVariable())},
		// NodeCos: cos(x) — siempre en [-1,1]
		{"cos(x)", expression.NewUnary(expression.NodeCos, expression.NewVariable())},
		// Composición: abs(sin(x)) — siempre válido
		{"abs(sin(x))", expression.NewUnary(expression.NodeAbs,
			expression.NewUnary(expression.NodeSin, expression.NewVariable()))},
		// Composición: sin(x^2)
		{"sin(x^2)", expression.NewUnary(expression.NodeSin,
			expression.NewBinary(expression.NodePow,
				expression.NewVariable(), expression.NewConstant(2)))},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := fitness.Evaluate(tc.expr, ds, 0.01)
			if err != nil {
				t.Fatalf("error inesperado: %v", err)
			}
			if math.IsNaN(result.MSE) || math.IsInf(result.MSE, 0) {
				t.Errorf("MSE no finito: %v", result.MSE)
			}
			if math.IsNaN(result.Fitness) || math.IsInf(result.Fitness, 0) {
				t.Errorf("Fitness no finito: %v", result.Fitness)
			}
			if result.MSE < 0 {
				t.Errorf("MSE negativo: %v", result.MSE)
			}
		})
	}
}

// ══════════════════════════════════════════════════════════════════════════
// 2. Operadores de Fase 8 — penalización cuando la expresión es inválida
// ══════════════════════════════════════════════════════════════════════════

func TestFitness_Phase8_InvalidOps_Penalized(t *testing.T) {
	// Dataset incluye x = 0 y negativos para activar dominios inválidos.
	ds := dataset.New([]dataset.Point{
		{X: -2, Y: 1}, {X: -1, Y: 1}, {X: 0, Y: 1}, {X: 1, Y: 1}, {X: 2, Y: 1},
	})

	cases := []struct {
		name          string
		expr          *expression.Node
		wantInvalid   int  // mínimo de puntos inválidos esperados
	}{
		// sqrt(x): x<0 inválido → 2 puntos inválidos (x=-2, x=-1)
		{"sqrt(x) con x negativo", expression.NewUnary(expression.NodeSqrt, expression.NewVariable()), 2},
		// ln(x): x≤0 inválido → 3 puntos inválidos (x=-2, x=-1, x=0)
		{"ln(x) con x≤0", expression.NewUnary(expression.NodeLn, expression.NewVariable()), 3},
		// x^(-1) con x=0 → 1 punto inválido
		{"x^-1 con x=0", expression.NewBinary(expression.NodePow,
			expression.NewVariable(), expression.NewConstant(-1)), 1},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := fitness.Evaluate(tc.expr, ds, 0)
			if err != nil {
				t.Fatalf("error inesperado: %v", err)
			}
			if result.InvalidCount < tc.wantInvalid {
				t.Errorf("InvalidCount = %d, want ≥ %d", result.InvalidCount, tc.wantInvalid)
			}
			if math.IsInf(result.MSE, 0) || math.IsNaN(result.MSE) {
				t.Errorf("MSE no finito a pesar de usar penalización finita: %v", result.MSE)
			}
			// Con puntos inválidos, el MSE debe ser alto (≥ PenaltyPerInvalidPoint/n)
			minExpectedMSE := fitness.PenaltyPerInvalidPoint * float64(tc.wantInvalid) / float64(ds.Len())
			if result.MSE < minExpectedMSE {
				t.Errorf("MSE = %v demasiado bajo para %d puntos inválidos (mín esperado: %v)",
					result.MSE, tc.wantInvalid, minExpectedMSE)
			}
		})
	}
}

// ══════════════════════════════════════════════════════════════════════════
// 3. Overflow en el error cuadrático
// ══════════════════════════════════════════════════════════════════════════

func TestFitness_SquaredErrorOverflow(t *testing.T) {
	// pred = MaxFloat64/2, Y = -MaxFloat64/2
	// diff = Y - pred ≈ -MaxFloat64
	// sq   = diff^2 ≈ MaxFloat64^2 → overflow a +Inf
	// El evaluador debe tratar este sq como punto inválido y usar PenaltyPerInvalidPoint.
	bigPred := math.MaxFloat64 / 2
	bigNeg := -math.MaxFloat64 / 2

	// Expresión constante = MaxFloat64/2; dataset con Y = -MaxFloat64/2
	expr := expression.NewConstant(bigPred)
	ds := dataset.New([]dataset.Point{
		{X: 1, Y: bigNeg},
		{X: 2, Y: bigNeg},
	})

	result, err := fitness.Evaluate(expr, ds, 0)
	if err != nil {
		t.Fatalf("error inesperado: %v", err)
	}

	// sq = (bigNeg - bigPred)^2 = (-MaxFloat64)^2 → +Inf → penalizado
	if result.InvalidCount == 0 {
		t.Error("se esperaban puntos penalizados por overflow en sq = diff^2, InvalidCount=0")
	}
	if math.IsInf(result.MSE, 0) || math.IsNaN(result.MSE) {
		t.Errorf("MSE no finito: %v (el overflow en sq debe ser penalizado, no propagado)", result.MSE)
	}
}

// ══════════════════════════════════════════════════════════════════════════
// 4. Expresiones con resultados grandes pero finitos
// ══════════════════════════════════════════════════════════════════════════

func TestFitness_LargeButFiniteResult(t *testing.T) {
	// exp(x) para x=10 → e^10 ≈ 22026: grande pero finito
	ds := positiveDS(5, func(x float64) float64 { return x })
	expr := expression.NewUnary(expression.NodeExp, expression.NewVariable())

	result, err := fitness.Evaluate(expr, ds, 0)
	if err != nil {
		t.Fatalf("error inesperado: %v", err)
	}
	if result.InvalidCount != 0 {
		t.Errorf("InvalidCount = %d, want 0 (exp válida para x>0)", result.InvalidCount)
	}
	if math.IsNaN(result.MSE) || math.IsInf(result.MSE, 0) {
		t.Errorf("MSE no finito: %v", result.MSE)
	}
	// MSE debe ser > 0 (exp(x) ≠ x en general)
	if result.MSE == 0 {
		t.Error("MSE = 0: exp(x) no debería coincidir exactamente con y=x")
	}
}

func TestFitness_SqrtX2Plus1_AlwaysValid(t *testing.T) {
	// sqrt(x^2+1): siempre válido porque x^2+1 >= 1 para todo x finito.
	// Con x ∈ [-50, 50] y y = sqrt(x^2+1), MSE debe ser 0.
	expr := expression.NewUnary(expression.NodeSqrt,
		expression.NewBinary(expression.NodeAdd,
			expression.NewBinary(expression.NodePow,
				expression.NewVariable(), expression.NewConstant(2)),
			expression.NewConstant(1)))

	ds := positiveDS(20, func(x float64) float64 {
		return math.Sqrt(x*x + 1)
	})
	// Re-generar dataset con x que incluya valores negativos
	pts := make([]dataset.Point, 21)
	for i := range pts {
		x := float64(i - 10) // x ∈ [-10, 10]
		pts[i] = dataset.Point{X: x, Y: math.Sqrt(x*x + 1)}
	}
	ds = dataset.New(pts)

	result, err := fitness.Evaluate(expr, ds, 0)
	if err != nil {
		t.Fatalf("error inesperado: %v", err)
	}
	if result.InvalidCount != 0 {
		t.Errorf("InvalidCount = %d, want 0 (sqrt(x^2+1) siempre válido)", result.InvalidCount)
	}
	if result.MSE > 1e-20 {
		t.Errorf("MSE = %v, want ≈ 0 (expresión exacta)", result.MSE)
	}
}

// ══════════════════════════════════════════════════════════════════════════
// 5. Constante no finita en el árbol — conducta nueva de Fase 9
// ══════════════════════════════════════════════════════════════════════════

func TestFitness_NonFiniteConstantInExpr(t *testing.T) {
	// Una expresión con NodeConstant(NaN) produce EvalError en cada punto.
	// El fitness debe usar PenaltyPerInvalidPoint para todos ellos.
	expr := expression.NewBinary(expression.NodeAdd,
		expression.NewVariable(),
		expression.NewConstant(math.NaN()))
	ds := simpleDS()

	result, err := fitness.Evaluate(expr, ds, 0)
	if err != nil {
		t.Fatalf("error inesperado: %v", err)
	}
	if result.InvalidCount != ds.Len() {
		t.Errorf("InvalidCount = %d, want %d (todos los puntos deben fallar por NaN)", result.InvalidCount, ds.Len())
	}
	if math.IsInf(result.MSE, 0) || math.IsNaN(result.MSE) {
		t.Errorf("MSE no finito: %v (la penalización debe ser finita)", result.MSE)
	}
	if result.MSE != fitness.PenaltyPerInvalidPoint {
		t.Errorf("MSE = %v, want %v (todos inválidos = PenaltyPerInvalidPoint)",
			result.MSE, fitness.PenaltyPerInvalidPoint)
	}
}

func TestFitness_InfConstant_AllInvalid(t *testing.T) {
	expr := expression.NewUnary(expression.NodeSin,
		expression.NewConstant(math.Inf(1)))
	ds := simpleDS()

	result, err := fitness.Evaluate(expr, ds, 0)
	if err != nil {
		t.Fatalf("error inesperado: %v", err)
	}
	if result.InvalidCount != ds.Len() {
		t.Errorf("InvalidCount = %d, want %d", result.InvalidCount, ds.Len())
	}
	if math.IsInf(result.MSE, 0) || math.IsNaN(result.MSE) {
		t.Errorf("MSE no finito: %v", result.MSE)
	}
}

// ══════════════════════════════════════════════════════════════════════════
// 6. Invariantes: MSE ≥ 0, Fitness ≥ 0, Complexity ≥ 1
// ══════════════════════════════════════════════════════════════════════════

func TestFitness_Invariants(t *testing.T) {
	ds := simpleDS()
	lambda := 0.01

	exprs := []*expression.Node{
		expression.NewVariable(),
		expression.NewConstant(3),
		expression.NewBinary(expression.NodeAdd, expression.NewVariable(), expression.NewConstant(1)),
		expression.NewBinary(expression.NodeMul, expression.NewConstant(2), expression.NewVariable()),
		expression.NewUnary(expression.NodeAbs, expression.NewVariable()),
		expression.NewUnary(expression.NodeSin, expression.NewVariable()),
		expression.NewBinary(expression.NodeDiv, expression.NewConstant(1), expression.NewConstant(0)),  // todos inválidos
		expression.NewUnary(expression.NodeSqrt, expression.NewConstant(-1)), // todos inválidos
	}

	for i, expr := range exprs {
		result, err := fitness.Evaluate(expr, ds, lambda)
		if err != nil {
			t.Errorf("expr[%d]: error inesperado: %v", i, err)
			continue
		}
		if result.MSE < 0 {
			t.Errorf("expr[%d]: MSE < 0: %v", i, result.MSE)
		}
		if result.Fitness < 0 {
			t.Errorf("expr[%d]: Fitness < 0: %v", i, result.Fitness)
		}
		if result.Complexity < 1 {
			t.Errorf("expr[%d]: Complexity < 1: %d", i, result.Complexity)
		}
		if math.IsNaN(result.MSE) || math.IsInf(result.MSE, 0) {
			t.Errorf("expr[%d]: MSE no finito: %v", i, result.MSE)
		}
		if math.IsNaN(result.Fitness) || math.IsInf(result.Fitness, 0) {
			t.Errorf("expr[%d]: Fitness no finito: %v", i, result.Fitness)
		}
	}
}

// ══════════════════════════════════════════════════════════════════════════
// 7. NoPanic: ninguna combinación de expresión válida o inválida causa panic
// ══════════════════════════════════════════════════════════════════════════

func TestFitness_Robustness_NoPanic(t *testing.T) {
	ds := dataset.New([]dataset.Point{
		{X: -5, Y: 1}, {X: -1, Y: 1}, {X: 0, Y: 1}, {X: 1, Y: 1}, {X: 5, Y: 1},
	})

	problematicExprs := []*expression.Node{
		nil,                                                                          // nil
		expression.NewConstant(math.NaN()),                                           // constante NaN
		expression.NewConstant(math.Inf(1)),                                          // constante Inf
		expression.NewBinary(expression.NodeDiv, expression.NewConstant(1), expression.NewConstant(0)),
		expression.NewUnary(expression.NodeSqrt, expression.NewVariable()),           // falla para x<0
		expression.NewUnary(expression.NodeLn, expression.NewVariable()),             // falla para x≤0
		expression.NewBinary(expression.NodePow, expression.NewVariable(), expression.NewConstant(-1)), // falla para x=0
		expression.NewUnary(expression.NodeExp, expression.NewConstant(1000)),        // overflow
		expression.NewBinary(expression.NodeMul,
			expression.NewConstant(math.MaxFloat64),
			expression.NewConstant(math.MaxFloat64)), // overflow
	}

	for i, expr := range problematicExprs {
		t.Run("", func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("expr[%d]: panic inesperado: %v", i, r)
				}
			}()
			_, _ = fitness.Evaluate(expr, ds, 0.01)
		})
	}
}
