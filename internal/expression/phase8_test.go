package expression_test

// phase8_test.go — Tests de Fase 8: NodePow, NodeAbs, NodeExp, NodeSin, NodeCos.
//
// Cubre:
//   - Evaluación correcta de cada operación (casos válidos)
//   - Casos inválidos específicos por operador
//   - Propagación de errores desde subárboles hijos
//   - Resultados NaN/Inf producen EvalError
//   - Depth, Count y Clone para los nuevos nodos
//   - Format para cada nuevo NodeType

import (
	"math"
	"testing"

	"github.com/LautiSeverino/symbolic-regression/internal/expression"
)

// ─── Helpers locales ──────────────────────────────────────────────────────

// approxEqual reporta si |a - b| < tol.
func approxEqual(a, b, tol float64) bool {
	return math.Abs(a-b) < tol
}

// divByZeroNode construye la expresión (1 / 0) — produce EvalError.
func divByZeroNode() *expression.Node {
	return expression.NewBinary(expression.NodeDiv,
		expression.NewConstant(1),
		expression.NewConstant(0),
	)
}

// ══════════════════════════════════════════════════════════════════════════
// NodePow
// ══════════════════════════════════════════════════════════════════════════

func TestEvaluate_Pow_Valid(t *testing.T) {
	tol := 1e-10
	cases := []struct {
		name string
		node *expression.Node
		x    float64
		want float64
	}{
		// Constantes con exponente entero positivo
		{"2^3", expression.NewBinary(expression.NodePow,
			expression.NewConstant(2), expression.NewConstant(3)), 0, 8},
		{"2^0", expression.NewBinary(expression.NodePow,
			expression.NewConstant(2), expression.NewConstant(0)), 0, 1},
		{"2^1", expression.NewBinary(expression.NodePow,
			expression.NewConstant(2), expression.NewConstant(1)), 0, 2},
		{"2^4", expression.NewBinary(expression.NodePow,
			expression.NewConstant(2), expression.NewConstant(4)), 0, 16},

		// Variable como base
		{"x^2 en x=3", expression.NewBinary(expression.NodePow,
			expression.NewVariable(), expression.NewConstant(2)), 3, 9},
		{"x^3 en x=2", expression.NewBinary(expression.NodePow,
			expression.NewVariable(), expression.NewConstant(3)), 2, 8},
		{"x^0 en x=999", expression.NewBinary(expression.NodePow,
			expression.NewVariable(), expression.NewConstant(0)), 999, 1},

		// Exponente negativo (base ≠ 0)
		{"2^-1", expression.NewBinary(expression.NodePow,
			expression.NewConstant(2), expression.NewConstant(-1)), 0, 0.5},
		{"4^-2", expression.NewBinary(expression.NodePow,
			expression.NewConstant(4), expression.NewConstant(-2)), 0, 0.0625},
		{"x^-1 en x=4", expression.NewBinary(expression.NodePow,
			expression.NewVariable(), expression.NewConstant(-1)), 4, 0.25},

		// Base negativa con exponente entero — válido en math.Pow
		{"(-2)^3", expression.NewBinary(expression.NodePow,
			expression.NewConstant(-2), expression.NewConstant(3)), 0, -8},
		{"(-2)^2", expression.NewBinary(expression.NodePow,
			expression.NewConstant(-2), expression.NewConstant(2)), 0, 4},
		{"(-3)^4", expression.NewBinary(expression.NodePow,
			expression.NewConstant(-3), expression.NewConstant(4)), 0, 81},

		// Exponente fraccionario en base positiva
		{"4^0.5", expression.NewBinary(expression.NodePow,
			expression.NewConstant(4), expression.NewConstant(0.5)), 0, 2},
		{"9^0.5", expression.NewBinary(expression.NodePow,
			expression.NewConstant(9), expression.NewConstant(0.5)), 0, 3},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := expression.Evaluate(tc.node, tc.x)
			if err != nil {
				t.Fatalf("error inesperado: %v", err)
			}
			if !approxEqual(got, tc.want, tol) {
				t.Errorf("got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestEvaluate_Pow_ZeroWithNegativeExponent(t *testing.T) {
	// 0^n con n < 0 → EvalError explícito
	cases := []struct {
		name string
		exp  float64
	}{
		{"0^-1", -1},
		{"0^-2", -2},
		{"0^-4", -4},
		{"0^-0.5", -0.5},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			node := expression.NewBinary(expression.NodePow,
				expression.NewConstant(0), expression.NewConstant(tc.exp))
			_, err := expression.Evaluate(node, 0)
			if err == nil {
				t.Errorf("0^%v: se esperaba EvalError, se obtuvo nil", tc.exp)
			}
		})
	}
}

func TestEvaluate_Pow_NegativeBaseNonIntegerExponent(t *testing.T) {
	// Base negativa con exponente no entero → NaN → EvalError
	cases := []struct {
		name string
		base float64
		exp  float64
	}{
		{"(-2)^0.5", -2, 0.5},
		{"(-3)^1.5", -3, 1.5},
		{"(-1)^0.1", -1, 0.1},
		{"(-4)^2.5", -4, 2.5},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			node := expression.NewBinary(expression.NodePow,
				expression.NewConstant(tc.base), expression.NewConstant(tc.exp))
			_, err := expression.Evaluate(node, 0)
			if err == nil {
				t.Errorf("(%v)^%v: se esperaba EvalError por NaN, se obtuvo nil", tc.base, tc.exp)
			}
		})
	}
}

func TestEvaluate_Pow_Overflow(t *testing.T) {
	// Base y exponente que producen +Inf por overflow
	node := expression.NewBinary(expression.NodePow,
		expression.NewConstant(math.MaxFloat64), expression.NewConstant(2))
	_, err := expression.Evaluate(node, 0)
	if err == nil {
		t.Error("MaxFloat64^2: se esperaba EvalError por overflow, se obtuvo nil")
	}
}

func TestEvaluate_Pow_PropagatesInnerErrorLeft(t *testing.T) {
	// (1/0)^2: el error del hijo izquierdo debe propagarse
	node := expression.NewBinary(expression.NodePow, divByZeroNode(), expression.NewConstant(2))
	_, err := expression.Evaluate(node, 0)
	if err == nil {
		t.Error("se esperaba EvalError propagado desde hijo izquierdo, se obtuvo nil")
	}
}

func TestEvaluate_Pow_PropagatesInnerErrorRight(t *testing.T) {
	// 2^(1/0): el error del hijo derecho debe propagarse
	node := expression.NewBinary(expression.NodePow, expression.NewConstant(2), divByZeroNode())
	_, err := expression.Evaluate(node, 0)
	if err == nil {
		t.Error("se esperaba EvalError propagado desde hijo derecho, se obtuvo nil")
	}
}

// ══════════════════════════════════════════════════════════════════════════
// NodeAbs
// ══════════════════════════════════════════════════════════════════════════

func TestEvaluate_Abs_Valid(t *testing.T) {
	cases := []struct {
		name string
		node *expression.Node
		x    float64
		want float64
	}{
		{"abs(3)", expression.NewUnary(expression.NodeAbs, expression.NewConstant(3)), 0, 3},
		{"abs(-3)", expression.NewUnary(expression.NodeAbs, expression.NewConstant(-3)), 0, 3},
		{"abs(0)", expression.NewUnary(expression.NodeAbs, expression.NewConstant(0)), 0, 0},
		{"abs(x) en x=5", expression.NewUnary(expression.NodeAbs, expression.NewVariable()), 5, 5},
		{"abs(x) en x=-5", expression.NewUnary(expression.NodeAbs, expression.NewVariable()), -5, 5},
		{"abs(x) en x=-0.7", expression.NewUnary(expression.NodeAbs, expression.NewVariable()), -0.7, 0.7},
		// abs de subexpresión: abs(x - 3) en x=1 → abs(-2) = 2
		{"abs(x-3) en x=1",
			expression.NewUnary(expression.NodeAbs,
				expression.NewBinary(expression.NodeSub,
					expression.NewVariable(), expression.NewConstant(3))),
			1, 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := expression.Evaluate(tc.node, tc.x)
			if err != nil {
				t.Fatalf("error inesperado: %v", err)
			}
			if got != tc.want {
				t.Errorf("got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestEvaluate_Abs_PropagatesInnerError(t *testing.T) {
	node := expression.NewUnary(expression.NodeAbs, divByZeroNode())
	_, err := expression.Evaluate(node, 0)
	if err == nil {
		t.Error("abs(1/0): se esperaba EvalError propagado, se obtuvo nil")
	}
}

// ══════════════════════════════════════════════════════════════════════════
// NodeExp
// ══════════════════════════════════════════════════════════════════════════

func TestEvaluate_Exp_Valid(t *testing.T) {
	tol := 1e-10
	cases := []struct {
		name string
		node *expression.Node
		x    float64
		want float64
	}{
		{"exp(0)=1", expression.NewUnary(expression.NodeExp, expression.NewConstant(0)), 0, 1},
		{"exp(1)=e", expression.NewUnary(expression.NodeExp, expression.NewConstant(1)), 0, math.E},
		{"exp(-1)=1/e", expression.NewUnary(expression.NodeExp, expression.NewConstant(-1)), 0, 1 / math.E},
		// exp(x) en x=0 → 1
		{"exp(x) en x=0", expression.NewUnary(expression.NodeExp, expression.NewVariable()), 0, 1},
		// exp(x) en x=2 → e²
		{"exp(x) en x=2", expression.NewUnary(expression.NodeExp, expression.NewVariable()), 2, math.E * math.E},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := expression.Evaluate(tc.node, tc.x)
			if err != nil {
				t.Fatalf("error inesperado: %v", err)
			}
			if !approxEqual(got, tc.want, tol) {
				t.Errorf("got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestEvaluate_Exp_Overflow(t *testing.T) {
	// exp(1000) → +Inf → EvalError
	node := expression.NewUnary(expression.NodeExp, expression.NewConstant(1000))
	_, err := expression.Evaluate(node, 0)
	if err == nil {
		t.Error("exp(1000): se esperaba EvalError por overflow, se obtuvo nil")
	}
}

func TestEvaluate_Exp_PropagatesInnerError(t *testing.T) {
	node := expression.NewUnary(expression.NodeExp, divByZeroNode())
	_, err := expression.Evaluate(node, 0)
	if err == nil {
		t.Error("exp(1/0): se esperaba EvalError propagado, se obtuvo nil")
	}
}

// ══════════════════════════════════════════════════════════════════════════
// NodeSin
// ══════════════════════════════════════════════════════════════════════════

func TestEvaluate_Sin_Valid(t *testing.T) {
	tol := 1e-10
	cases := []struct {
		name string
		node *expression.Node
		x    float64
		want float64
	}{
		{"sin(0)=0", expression.NewUnary(expression.NodeSin, expression.NewConstant(0)), 0, 0},
		{"sin(π/2)≈1", expression.NewUnary(expression.NodeSin, expression.NewConstant(math.Pi/2)), 0, 1},
		{"sin(π)≈0", expression.NewUnary(expression.NodeSin, expression.NewConstant(math.Pi)), 0, 0},
		{"sin(-π/2)≈-1", expression.NewUnary(expression.NodeSin, expression.NewConstant(-math.Pi/2)), 0, -1},
		// sin(x) en x=0 → 0
		{"sin(x) en x=0", expression.NewUnary(expression.NodeSin, expression.NewVariable()), 0, 0},
		// sin(x) en x=π/2 → 1
		{"sin(x) en x=π/2", expression.NewUnary(expression.NodeSin, expression.NewVariable()), math.Pi / 2, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := expression.Evaluate(tc.node, tc.x)
			if err != nil {
				t.Fatalf("error inesperado: %v", err)
			}
			if !approxEqual(got, tc.want, tol) {
				t.Errorf("got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestEvaluate_Sin_ResultBounded(t *testing.T) {
	// sin siempre produce valores en [-1, 1] para argumentos finitos
	xValues := []float64{-1e6, -100, -1, 0, 1, 100, 1e6}
	node := expression.NewUnary(expression.NodeSin, expression.NewVariable())
	for _, x := range xValues {
		got, err := expression.Evaluate(node, x)
		if err != nil {
			t.Errorf("x=%v: error inesperado: %v", x, err)
			continue
		}
		if got < -1-1e-12 || got > 1+1e-12 {
			t.Errorf("x=%v: sin fuera de [-1,1]: %v", x, got)
		}
	}
}

func TestEvaluate_Sin_PropagatesInnerError(t *testing.T) {
	node := expression.NewUnary(expression.NodeSin, divByZeroNode())
	_, err := expression.Evaluate(node, 0)
	if err == nil {
		t.Error("sin(1/0): se esperaba EvalError propagado, se obtuvo nil")
	}
}

// ══════════════════════════════════════════════════════════════════════════
// NodeCos
// ══════════════════════════════════════════════════════════════════════════

func TestEvaluate_Cos_Valid(t *testing.T) {
	tol := 1e-10
	cases := []struct {
		name string
		node *expression.Node
		x    float64
		want float64
	}{
		{"cos(0)=1", expression.NewUnary(expression.NodeCos, expression.NewConstant(0)), 0, 1},
		{"cos(π)≈-1", expression.NewUnary(expression.NodeCos, expression.NewConstant(math.Pi)), 0, -1},
		{"cos(π/2)≈0", expression.NewUnary(expression.NodeCos, expression.NewConstant(math.Pi/2)), 0, 0},
		{"cos(-π)≈-1", expression.NewUnary(expression.NodeCos, expression.NewConstant(-math.Pi)), 0, -1},
		// cos(x) en x=0 → 1
		{"cos(x) en x=0", expression.NewUnary(expression.NodeCos, expression.NewVariable()), 0, 1},
		// cos(x) en x=π → -1
		{"cos(x) en x=π", expression.NewUnary(expression.NodeCos, expression.NewVariable()), math.Pi, -1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := expression.Evaluate(tc.node, tc.x)
			if err != nil {
				t.Fatalf("error inesperado: %v", err)
			}
			if !approxEqual(got, tc.want, tol) {
				t.Errorf("got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestEvaluate_Cos_ResultBounded(t *testing.T) {
	xValues := []float64{-1e6, -100, -1, 0, 1, 100, 1e6}
	node := expression.NewUnary(expression.NodeCos, expression.NewVariable())
	for _, x := range xValues {
		got, err := expression.Evaluate(node, x)
		if err != nil {
			t.Errorf("x=%v: error inesperado: %v", x, err)
			continue
		}
		if got < -1-1e-12 || got > 1+1e-12 {
			t.Errorf("x=%v: cos fuera de [-1,1]: %v", x, got)
		}
	}
}

func TestEvaluate_Cos_PropagatesInnerError(t *testing.T) {
	node := expression.NewUnary(expression.NodeCos, divByZeroNode())
	_, err := expression.Evaluate(node, 0)
	if err == nil {
		t.Error("cos(1/0): se esperaba EvalError propagado, se obtuvo nil")
	}
}

// ══════════════════════════════════════════════════════════════════════════
// NoPanic — ningún nuevo operador produce panic
// ══════════════════════════════════════════════════════════════════════════

func TestEvaluate_Phase8_NoPanic(t *testing.T) {
	ds := []float64{-1e10, -100, -2, -1, -0.5, 0, 0.5, 1, 2, 100, 1e10}
	cases := []struct {
		name string
		node *expression.Node
	}{
		{"x^2", expression.NewBinary(expression.NodePow,
			expression.NewVariable(), expression.NewConstant(2))},
		{"x^-1", expression.NewBinary(expression.NodePow,
			expression.NewVariable(), expression.NewConstant(-1))},
		{"0^-1 (const)", expression.NewBinary(expression.NodePow,
			expression.NewConstant(0), expression.NewConstant(-1))},
		{"(-2)^0.5", expression.NewBinary(expression.NodePow,
			expression.NewConstant(-2), expression.NewConstant(0.5))},
		{"exp(1000)", expression.NewUnary(expression.NodeExp, expression.NewConstant(1000))},
		{"abs(x)", expression.NewUnary(expression.NodeAbs, expression.NewVariable())},
		{"sin(x)", expression.NewUnary(expression.NodeSin, expression.NewVariable())},
		{"cos(x)", expression.NewUnary(expression.NodeCos, expression.NewVariable())},
		{"nilNode", nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for _, x := range ds {
				defer func() {
					if r := recover(); r != nil {
						t.Errorf("panic en %s con x=%v: %v", tc.name, x, r)
					}
				}()
				_, _ = expression.Evaluate(tc.node, x)
			}
		})
	}
}

// ══════════════════════════════════════════════════════════════════════════
// Estructura: Depth y Count
// ══════════════════════════════════════════════════════════════════════════

func TestDepth_Phase8(t *testing.T) {
	cases := []struct {
		name string
		node *expression.Node
		want int
	}{
		// Hojas: depth = 1
		{"NodePow es binario — hoja no aplica", nil, 0},

		// x^2: NodePow(NodeVariable, NodeConstant) → depth 2
		{"x^2", expression.NewBinary(expression.NodePow,
			expression.NewVariable(), expression.NewConstant(2)), 2},

		// abs(x): depth 2
		{"abs(x)", expression.NewUnary(expression.NodeAbs, expression.NewVariable()), 2},
		// exp(x): depth 2
		{"exp(x)", expression.NewUnary(expression.NodeExp, expression.NewVariable()), 2},
		// sin(x): depth 2
		{"sin(x)", expression.NewUnary(expression.NodeSin, expression.NewVariable()), 2},
		// cos(x): depth 2
		{"cos(x)", expression.NewUnary(expression.NodeCos, expression.NewVariable()), 2},

		// exp(x + 1): NodeExp(NodeAdd(NodeVariable, NodeConstant)) → depth 3
		{"exp(x+1)", expression.NewUnary(expression.NodeExp,
			expression.NewBinary(expression.NodeAdd,
				expression.NewVariable(), expression.NewConstant(1))), 3},

		// abs(x^2): NodeAbs(NodePow(NodeVariable, NodeConstant)) → depth 3
		{"abs(x^2)", expression.NewUnary(expression.NodeAbs,
			expression.NewBinary(expression.NodePow,
				expression.NewVariable(), expression.NewConstant(2))), 3},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := expression.Depth(tc.node)
			if got != tc.want {
				t.Errorf("Depth = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestCount_Phase8(t *testing.T) {
	cases := []struct {
		name string
		node *expression.Node
		want int
	}{
		// x^2: NodePow + NodeVariable + NodeConstant(2) = 3
		{"x^2", expression.NewBinary(expression.NodePow,
			expression.NewVariable(), expression.NewConstant(2)), 3},

		// abs(x): NodeAbs + NodeVariable = 2
		{"abs(x)", expression.NewUnary(expression.NodeAbs, expression.NewVariable()), 2},
		// exp(x): 2
		{"exp(x)", expression.NewUnary(expression.NodeExp, expression.NewVariable()), 2},
		// sin(x): 2
		{"sin(x)", expression.NewUnary(expression.NodeSin, expression.NewVariable()), 2},
		// cos(x): 2
		{"cos(x)", expression.NewUnary(expression.NodeCos, expression.NewVariable()), 2},

		// sin(x + 2): NodeSin + NodeAdd + NodeVariable + NodeConstant = 4
		{"sin(x+2)", expression.NewUnary(expression.NodeSin,
			expression.NewBinary(expression.NodeAdd,
				expression.NewVariable(), expression.NewConstant(2))), 4},

		// abs(x^2): NodeAbs + NodePow + NodeVariable + NodeConstant = 4
		{"abs(x^2)", expression.NewUnary(expression.NodeAbs,
			expression.NewBinary(expression.NodePow,
				expression.NewVariable(), expression.NewConstant(2))), 4},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := expression.Count(tc.node)
			if got != tc.want {
				t.Errorf("Count = %d, want %d", got, tc.want)
			}
		})
	}
}

// ══════════════════════════════════════════════════════════════════════════
// Clone — independencia para los nuevos nodos
// ══════════════════════════════════════════════════════════════════════════

func TestClone_Phase8_BinaryPow(t *testing.T) {
	// Clone de x^2: modificar el clon no debe afectar al original
	original := expression.NewBinary(expression.NodePow,
		expression.NewVariable(), expression.NewConstant(2))
	clone := expression.Clone(original)

	// Modificar el exponente del clon
	clone.Right.Value = 999

	if original.Right.Value != 2 {
		t.Errorf("modificar clone.Right.Value afectó original: got %v, want 2", original.Right.Value)
	}
}

func TestClone_Phase8_Unary(t *testing.T) {
	// Clone de abs(sin(x)): modificar en profundidad no afecta el original
	original := expression.NewUnary(expression.NodeAbs,
		expression.NewUnary(expression.NodeSin, expression.NewVariable()))
	clone := expression.Clone(original)

	clone.Left.Type = expression.NodeCos

	if original.Left.Type != expression.NodeSin {
		t.Errorf("modificar clone.Left.Type afectó original: got %v, want NodeSin", original.Left.Type)
	}
}

func TestClone_Phase8_NilRight(t *testing.T) {
	// Nodos unarios siempre deben tener Right == nil en original y clon
	unaries := []*expression.Node{
		expression.NewUnary(expression.NodeAbs, expression.NewVariable()),
		expression.NewUnary(expression.NodeExp, expression.NewVariable()),
		expression.NewUnary(expression.NodeSin, expression.NewVariable()),
		expression.NewUnary(expression.NodeCos, expression.NewVariable()),
	}
	for _, u := range unaries {
		c := expression.Clone(u)
		if c.Right != nil {
			t.Errorf("NodeType %d: clone.Right ≠ nil (debe ser nil para nodo unario)", u.Type)
		}
	}
}

// ══════════════════════════════════════════════════════════════════════════
// Format — representación de cada nuevo NodeType
// ══════════════════════════════════════════════════════════════════════════

func TestFormat_Phase8(t *testing.T) {
	cases := []struct {
		name   string
		node   *expression.Node
		expect string
	}{
		// NodePow — binario parentesizado
		{"x^2", expression.NewBinary(expression.NodePow,
			expression.NewVariable(), expression.NewConstant(2)),
			"(x ^ 2)"},
		{"x^-1", expression.NewBinary(expression.NodePow,
			expression.NewVariable(), expression.NewConstant(-1)),
			"(x ^ -1)"},
		{"(x+1)^3", expression.NewBinary(expression.NodePow,
			expression.NewBinary(expression.NodeAdd,
				expression.NewVariable(), expression.NewConstant(1)),
			expression.NewConstant(3)),
			"((x + 1) ^ 3)"},

		// NodeAbs — unario
		{"abs(x)", expression.NewUnary(expression.NodeAbs, expression.NewVariable()),
			"abs(x)"},
		{"abs(x-3)", expression.NewUnary(expression.NodeAbs,
			expression.NewBinary(expression.NodeSub,
				expression.NewVariable(), expression.NewConstant(3))),
			"abs((x - 3))"},

		// NodeExp — unario
		{"exp(x)", expression.NewUnary(expression.NodeExp, expression.NewVariable()),
			"exp(x)"},
		{"exp(x+1)", expression.NewUnary(expression.NodeExp,
			expression.NewBinary(expression.NodeAdd,
				expression.NewVariable(), expression.NewConstant(1))),
			"exp((x + 1))"},

		// NodeSin — unario
		{"sin(x)", expression.NewUnary(expression.NodeSin, expression.NewVariable()),
			"sin(x)"},
		{"sin(x*2)", expression.NewUnary(expression.NodeSin,
			expression.NewBinary(expression.NodeMul,
				expression.NewVariable(), expression.NewConstant(2))),
			"sin((x * 2))"},

		// NodeCos — unario
		{"cos(x)", expression.NewUnary(expression.NodeCos, expression.NewVariable()),
			"cos(x)"},
		{"cos(-1)", expression.NewUnary(expression.NodeCos, expression.NewConstant(-1)),
			"cos(-1)"},

		// Composición: sin dentro de cos (aunque el generador no la produzca por defecto)
		{"sin(cos(x))", expression.NewUnary(expression.NodeSin,
			expression.NewUnary(expression.NodeCos, expression.NewVariable())),
			"sin(cos(x))"},

		// abs(x^2) — composición de NodeAbs y NodePow
		{"abs(x^2)", expression.NewUnary(expression.NodeAbs,
			expression.NewBinary(expression.NodePow,
				expression.NewVariable(), expression.NewConstant(2))),
			"abs((x ^ 2))"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := expression.Format(tc.node)
			if got != tc.expect {
				t.Errorf("got %q, want %q", got, tc.expect)
			}
		})
	}
}

// ══════════════════════════════════════════════════════════════════════════
// Regresión — verificar que Fases 1-7 no se rompieron
// ══════════════════════════════════════════════════════════════════════════

func TestRegression_Phase8_ExistingNodeTypes(t *testing.T) {
	// Los iota de los NodeType existentes no deben haber cambiado.
	// Se comprueban directamente sus valores.
	if expression.NodeVariable != 0 {
		t.Errorf("NodeVariable = %d, want 0", expression.NodeVariable)
	}
	if expression.NodeConstant != 1 {
		t.Errorf("NodeConstant = %d, want 1", expression.NodeConstant)
	}
	if expression.NodeAdd != 2 {
		t.Errorf("NodeAdd = %d, want 2", expression.NodeAdd)
	}
	if expression.NodeSub != 3 {
		t.Errorf("NodeSub = %d, want 3", expression.NodeSub)
	}
	if expression.NodeMul != 4 {
		t.Errorf("NodeMul = %d, want 4", expression.NodeMul)
	}
	if expression.NodeDiv != 5 {
		t.Errorf("NodeDiv = %d, want 5", expression.NodeDiv)
	}
	if expression.NodeSqrt != 6 {
		t.Errorf("NodeSqrt = %d, want 6", expression.NodeSqrt)
	}
	if expression.NodeLn != 7 {
		t.Errorf("NodeLn = %d, want 7", expression.NodeLn)
	}
	// Nuevos
	if expression.NodePow != 8 {
		t.Errorf("NodePow = %d, want 8", expression.NodePow)
	}
	if expression.NodeAbs != 9 {
		t.Errorf("NodeAbs = %d, want 9", expression.NodeAbs)
	}
	if expression.NodeExp != 10 {
		t.Errorf("NodeExp = %d, want 10", expression.NodeExp)
	}
	if expression.NodeSin != 11 {
		t.Errorf("NodeSin = %d, want 11", expression.NodeSin)
	}
	if expression.NodeCos != 12 {
		t.Errorf("NodeCos = %d, want 12", expression.NodeCos)
	}
}

func TestRegression_Phase8_Linear2x1(t *testing.T) {
	// El ejemplo canónico 2*x+1 debe seguir evaluando correctamente
	tree := expression.NewBinary(expression.NodeAdd,
		expression.NewBinary(expression.NodeMul,
			expression.NewConstant(2), expression.NewVariable()),
		expression.NewConstant(1))

	cases := [][2]float64{{1, 3}, {2, 5}, {3, 7}, {4, 9}, {5, 11}}
	for _, c := range cases {
		got, err := expression.Evaluate(tree, c[0])
		if err != nil {
			t.Fatalf("x=%.0f: error inesperado: %v", c[0], err)
		}
		if got != c[1] {
			t.Errorf("x=%.0f: got %v, want %v", c[0], got, c[1])
		}
	}
}
