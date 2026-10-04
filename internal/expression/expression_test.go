package expression_test

import (
	"math"
	"testing"

	"github.com/LautiSeverino/symbolic-regression/internal/expression"
)

// linear construye (2 * x) + 1 — el ejemplo canónico del spec.
// Dataset: x=1→3, x=2→5, x=3→7, x=4→9, x=5→11
func linear() *expression.Node {
	return expression.NewBinary(expression.NodeAdd,
		expression.NewBinary(expression.NodeMul,
			expression.NewConstant(2),
			expression.NewVariable(),
		),
		expression.NewConstant(1),
	)
}

// ─── Evaluate ──────────────────────────────────────────────────────────────

func TestEvaluate_Variable(t *testing.T) {
	node := expression.NewVariable()
	for _, x := range []float64{-5, 0, 3.14, 100} {
		got, err := expression.Evaluate(node, x)
		if err != nil {
			t.Fatalf("x=%v: error inesperado: %v", x, err)
		}
		if got != x {
			t.Errorf("x=%v: got %v, want %v", x, got, x)
		}
	}
}

func TestEvaluate_Constant(t *testing.T) {
	node := expression.NewConstant(3.14)
	for _, x := range []float64{-5, 0, 42} {
		got, err := expression.Evaluate(node, x)
		if err != nil {
			t.Fatalf("x=%v: error inesperado: %v", x, err)
		}
		if got != 3.14 {
			t.Errorf("x=%v: got %v, want 3.14", x, got)
		}
	}
}

// TestEvaluate_LinearExample cubre el dataset canónico del spec:
// f(x) = 2*x + 1 → {(1,3), (2,5), (3,7), (4,9), (5,11)}
func TestEvaluate_LinearExample(t *testing.T) {
	tree := linear()

	tests := []struct{ x, want float64 }{
		{1, 3},
		{2, 5},
		{3, 7},
		{4, 9},
		{5, 11},
	}
	for _, tt := range tests {
		got, err := expression.Evaluate(tree, tt.x)
		if err != nil {
			t.Fatalf("x=%.0f: error inesperado: %v", tt.x, err)
		}
		if got != tt.want {
			t.Errorf("x=%.0f: got %v, want %v", tt.x, got, tt.want)
		}
	}
}

func TestEvaluate_Subtraction(t *testing.T) {
	// x - 3 en x=10 → 7
	node := expression.NewBinary(expression.NodeSub, expression.NewVariable(), expression.NewConstant(3))
	got, err := expression.Evaluate(node, 10)
	if err != nil {
		t.Fatalf("error inesperado: %v", err)
	}
	if got != 7 {
		t.Errorf("got %v, want 7", got)
	}
}

func TestEvaluate_NestedExpr(t *testing.T) {
	// (x + 1) * 2
	inner := expression.NewBinary(expression.NodeAdd, expression.NewVariable(), expression.NewConstant(1))
	tree := expression.NewBinary(expression.NodeMul, inner, expression.NewConstant(2))

	tests := []struct{ x, want float64 }{
		{0, 2},
		{1, 4},
		{2, 6},
		{5, 12},
	}
	for _, tt := range tests {
		got, err := expression.Evaluate(tree, tt.x)
		if err != nil {
			t.Fatalf("x=%.0f: error inesperado: %v", tt.x, err)
		}
		if got != tt.want {
			t.Errorf("x=%.0f: got %v, want %v", tt.x, got, tt.want)
		}
	}
}

func TestEvaluate_Division(t *testing.T) {
	// x / 4 en x=8 → 2
	node := expression.NewBinary(expression.NodeDiv, expression.NewVariable(), expression.NewConstant(4))
	got, err := expression.Evaluate(node, 8)
	if err != nil {
		t.Fatalf("error inesperado: %v", err)
	}
	if got != 2 {
		t.Errorf("got %v, want 2", got)
	}
}

func TestEvaluate_DivisionByZeroConstant(t *testing.T) {
	// 1 / 0 — denominador constante cero
	node := expression.NewBinary(expression.NodeDiv,
		expression.NewConstant(1),
		expression.NewConstant(0),
	)
	_, err := expression.Evaluate(node, 0)
	if err == nil {
		t.Error("esperaba EvalError para división por cero, obtuve nil")
	}
}

func TestEvaluate_DivisionByZeroAtPoint(t *testing.T) {
	// x / (x - 5): denominador = 0 cuando x = 5
	node := expression.NewBinary(expression.NodeDiv,
		expression.NewVariable(),
		expression.NewBinary(expression.NodeSub,
			expression.NewVariable(),
			expression.NewConstant(5),
		),
	)

	// Debe fallar en x=5
	_, err := expression.Evaluate(node, 5)
	if err == nil {
		t.Error("x=5: esperaba EvalError por división por cero, obtuve nil")
	}

	// Debe funcionar en x=10 → 10/(10-5) = 2
	got, err := expression.Evaluate(node, 10)
	if err != nil {
		t.Fatalf("x=10: error inesperado: %v", err)
	}
	if got != 2 {
		t.Errorf("x=10: got %v, want 2", got)
	}
}

func TestEvaluate_NilNode(t *testing.T) {
	_, err := expression.Evaluate(nil, 1)
	if err == nil {
		t.Error("esperaba error para nodo nil, obtuve nil")
	}
}

func TestEvaluate_Overflow(t *testing.T) {
	// MaxFloat64 * MaxFloat64 → +Inf → EvalError
	node := expression.NewBinary(expression.NodeMul,
		expression.NewConstant(math.MaxFloat64),
		expression.NewConstant(math.MaxFloat64),
	)
	_, err := expression.Evaluate(node, 0)
	if err == nil {
		t.Error("esperaba EvalError por overflow en multiplicación, obtuve nil")
	}
}

// ─── Depth ────────────────────────────────────────────────────────────────

func TestDepth(t *testing.T) {
	tests := []struct {
		name string
		node *expression.Node
		want int
	}{
		{"nil", nil, 0},
		{"hoja variable", expression.NewVariable(), 1},
		{"hoja constante", expression.NewConstant(3), 1},
		{
			"x + 2 (profundidad 2)",
			expression.NewBinary(expression.NodeAdd,
				expression.NewVariable(), expression.NewConstant(2)),
			2,
		},
		{
			"(x + 1) * 2 (profundidad 3)",
			expression.NewBinary(expression.NodeMul,
				expression.NewBinary(expression.NodeAdd,
					expression.NewVariable(), expression.NewConstant(1)),
				expression.NewConstant(2)),
			3,
		},
		{"linear 2*x+1 (profundidad 3)", linear(), 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := expression.Depth(tt.node)
			if got != tt.want {
				t.Errorf("got %d, want %d", got, tt.want)
			}
		})
	}
}

func TestDepth_LeftHeavy(t *testing.T) {
	// ((x + 1) + 2) + 3 → árbol sesgado a la izquierda, profundidad 4
	t1 := expression.NewBinary(expression.NodeAdd, expression.NewVariable(), expression.NewConstant(1))
	t2 := expression.NewBinary(expression.NodeAdd, t1, expression.NewConstant(2))
	t3 := expression.NewBinary(expression.NodeAdd, t2, expression.NewConstant(3))
	got := expression.Depth(t3)
	if got != 4 {
		t.Errorf("árbol izquierdo: got depth %d, want 4", got)
	}
}

// ─── Count ────────────────────────────────────────────────────────────────

func TestCount(t *testing.T) {
	tests := []struct {
		name string
		node *expression.Node
		want int
	}{
		{"nil", nil, 0},
		{"hoja única", expression.NewVariable(), 1},
		{
			"x + 2 (3 nodos)",
			expression.NewBinary(expression.NodeAdd,
				expression.NewVariable(), expression.NewConstant(2)),
			3,
		},
		{
			"(x+1)*2 (5 nodos)",
			expression.NewBinary(expression.NodeMul,
				expression.NewBinary(expression.NodeAdd,
					expression.NewVariable(), expression.NewConstant(1)),
				expression.NewConstant(2)),
			5,
		},
		{"linear 2*x+1 (5 nodos)", linear(), 5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := expression.Count(tt.node)
			if got != tt.want {
				t.Errorf("got %d, want %d", got, tt.want)
			}
		})
	}
}

// ─── Clone ────────────────────────────────────────────────────────────────

func TestClone_IsIndependent(t *testing.T) {
	original := expression.NewBinary(expression.NodeMul,
		expression.NewVariable(),
		expression.NewConstant(3),
	)
	clone := expression.Clone(original)

	// Modificar el clon no debe afectar al original
	clone.Left = expression.NewConstant(99)

	if original.Left.Type != expression.NodeVariable {
		t.Error("modificar clone.Left afectó original.Left")
	}
}

func TestClone_DeepIndependence(t *testing.T) {
	// Árbol de profundidad 3: (x + 1) * 3
	original := expression.NewBinary(expression.NodeMul,
		expression.NewBinary(expression.NodeAdd,
			expression.NewVariable(), expression.NewConstant(1)),
		expression.NewConstant(3),
	)
	clone := expression.Clone(original)

	// Modificar un nodo profundo del clon
	clone.Left.Right.Value = 999

	if original.Left.Right.Value != 1 {
		t.Errorf("modificación profunda del clon afectó original: got %v, want 1",
			original.Left.Right.Value)
	}
}

func TestClone_PreservesValues(t *testing.T) {
	original := expression.NewConstant(42.5)
	clone := expression.Clone(original)

	if clone.Value != 42.5 {
		t.Errorf("clone.Value = %v, want 42.5", clone.Value)
	}
	if clone == original {
		t.Error("clone y original son el mismo puntero")
	}
}

func TestClone_NilIsNil(t *testing.T) {
	if expression.Clone(nil) != nil {
		t.Error("Clone(nil) debe retornar nil")
	}
}

// ─── Format ───────────────────────────────────────────────────────────────

func TestFormat(t *testing.T) {
	tests := []struct {
		name   string
		node   *expression.Node
		expect string
	}{
		{"nil", nil, "<nil>"},
		{"variable", expression.NewVariable(), "x"},
		{"entero positivo", expression.NewConstant(3), "3"},
		{"entero negativo", expression.NewConstant(-2), "-2"},
		{"float", expression.NewConstant(1.5), "1.5"},
		{
			"x + 2",
			expression.NewBinary(expression.NodeAdd,
				expression.NewVariable(), expression.NewConstant(2)),
			"(x + 2)",
		},
		{
			"x * 3",
			expression.NewBinary(expression.NodeMul,
				expression.NewVariable(), expression.NewConstant(3)),
			"(x * 3)",
		},
		{
			"(x + 1) * 2",
			expression.NewBinary(expression.NodeMul,
				expression.NewBinary(expression.NodeAdd,
					expression.NewVariable(), expression.NewConstant(1)),
				expression.NewConstant(2)),
			"((x + 1) * 2)",
		},
		{
			"linear 2*x+1",
			linear(),
			"((2 * x) + 1)",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := expression.Format(tt.node)
			if got != tt.expect {
				t.Errorf("got %q, want %q", got, tt.expect)
			}
		})
	}
}

// ─── Sqrt — Fase 2 ────────────────────────────────────────────────────────

func TestEvaluate_Sqrt_Valid(t *testing.T) {
	tests := []struct {
		name string
		node *expression.Node
		x    float64
		want float64
	}{
		// sqrt de constante conocida
		{"sqrt(0)", expression.NewUnary(expression.NodeSqrt, expression.NewConstant(0)), 0, 0},
		{"sqrt(1)", expression.NewUnary(expression.NodeSqrt, expression.NewConstant(1)), 0, 1},
		{"sqrt(4)", expression.NewUnary(expression.NodeSqrt, expression.NewConstant(4)), 0, 2},
		{"sqrt(9)", expression.NewUnary(expression.NodeSqrt, expression.NewConstant(9)), 0, 3},
		// sqrt de variable
		{"sqrt(x) en x=4", expression.NewUnary(expression.NodeSqrt, expression.NewVariable()), 4, 2},
		{"sqrt(x) en x=9", expression.NewUnary(expression.NodeSqrt, expression.NewVariable()), 9, 3},
		// sqrt de subexpresión: sqrt(x + 7) en x=2 → sqrt(9) = 3
		{
			"sqrt(x+7) en x=2",
			expression.NewUnary(expression.NodeSqrt,
				expression.NewBinary(expression.NodeAdd,
					expression.NewVariable(), expression.NewConstant(7))),
			2, 3,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := expression.Evaluate(tt.node, tt.x)
			if err != nil {
				t.Fatalf("error inesperado: %v", err)
			}
			if got != tt.want {
				t.Errorf("got %v, want %v", got, tt.want)
			}
		})
	}
}

func TestEvaluate_Sqrt_Negative(t *testing.T) {
	cases := []struct {
		name string
		node *expression.Node
		x    float64
	}{
		{"sqrt(-1)", expression.NewUnary(expression.NodeSqrt, expression.NewConstant(-1)), 0},
		{"sqrt(-0.001)", expression.NewUnary(expression.NodeSqrt, expression.NewConstant(-0.001)), 0},
		{"sqrt(x) en x=-4", expression.NewUnary(expression.NodeSqrt, expression.NewVariable()), -4},
		{"sqrt(x) en x=-0.1", expression.NewUnary(expression.NodeSqrt, expression.NewVariable()), -0.1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := expression.Evaluate(tc.node, tc.x)
			if err == nil {
				t.Error("esperaba EvalError para sqrt de negativo, obtuve nil")
			}
		})
	}
}

func TestEvaluate_Sqrt_PropagatesInnerError(t *testing.T) {
	// sqrt(1 / 0): el error del hijo debe propagarse
	inner := expression.NewBinary(expression.NodeDiv,
		expression.NewConstant(1),
		expression.NewConstant(0),
	)
	node := expression.NewUnary(expression.NodeSqrt, inner)
	_, err := expression.Evaluate(node, 0)
	if err == nil {
		t.Error("esperaba EvalError propagado desde el hijo, obtuve nil")
	}
}

// ─── Ln — Fase 2 ──────────────────────────────────────────────────────────

func TestEvaluate_Ln_Valid(t *testing.T) {
	tests := []struct {
		name string
		node *expression.Node
		x    float64
		want float64
	}{
		// ln(1) = 0 exacto
		{"ln(1)", expression.NewUnary(expression.NodeLn, expression.NewConstant(1)), 0, 0},
		// ln(x) en x=1 → 0 exacto
		{"ln(x) en x=1", expression.NewUnary(expression.NodeLn, expression.NewVariable()), 1, 0},
		// ln(x+1) en x=0 → ln(1) = 0
		{
			"ln(x+1) en x=0",
			expression.NewUnary(expression.NodeLn,
				expression.NewBinary(expression.NodeAdd,
					expression.NewVariable(), expression.NewConstant(1))),
			0, 0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := expression.Evaluate(tt.node, tt.x)
			if err != nil {
				t.Fatalf("error inesperado: %v", err)
			}
			if got != tt.want {
				t.Errorf("got %v, want %v", got, tt.want)
			}
		})
	}
}

func TestEvaluate_Ln_Invalid(t *testing.T) {
	cases := []struct {
		name string
		node *expression.Node
		x    float64
	}{
		// argumento cero
		{"ln(0)", expression.NewUnary(expression.NodeLn, expression.NewConstant(0)), 0},
		{"ln(x) en x=0", expression.NewUnary(expression.NodeLn, expression.NewVariable()), 0},
		// argumento negativo
		{"ln(-1)", expression.NewUnary(expression.NodeLn, expression.NewConstant(-1)), 0},
		{"ln(-100)", expression.NewUnary(expression.NodeLn, expression.NewConstant(-100)), 0},
		{"ln(x) en x=-3", expression.NewUnary(expression.NodeLn, expression.NewVariable()), -3},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := expression.Evaluate(tc.node, tc.x)
			if err == nil {
				t.Errorf("esperaba EvalError para ln de no-positivo, obtuve nil")
			}
		})
	}
}

func TestEvaluate_Ln_PropagatesInnerError(t *testing.T) {
	// ln(x / 0): el error del hijo debe propagarse
	inner := expression.NewBinary(expression.NodeDiv,
		expression.NewVariable(),
		expression.NewConstant(0),
	)
	node := expression.NewUnary(expression.NodeLn, inner)
	_, err := expression.Evaluate(node, 5)
	if err == nil {
		t.Error("esperaba EvalError propagado desde el hijo, obtuve nil")
	}
}

// ─── Estructura de nodos unarios ──────────────────────────────────────────

func TestDepth_Unary(t *testing.T) {
	tests := []struct {
		name string
		node *expression.Node
		want int
	}{
		{
			// sqrt(x) → profundidad 2
			"sqrt(x)",
			expression.NewUnary(expression.NodeSqrt, expression.NewVariable()),
			2,
		},
		{
			// ln(x) → profundidad 2
			"ln(x)",
			expression.NewUnary(expression.NodeLn, expression.NewVariable()),
			2,
		},
		{
			// sqrt(x + 1) → profundidad 3
			"sqrt(x+1)",
			expression.NewUnary(expression.NodeSqrt,
				expression.NewBinary(expression.NodeAdd,
					expression.NewVariable(), expression.NewConstant(1))),
			3,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := expression.Depth(tt.node)
			if got != tt.want {
				t.Errorf("got %d, want %d", got, tt.want)
			}
		})
	}
}

func TestCount_Unary(t *testing.T) {
	tests := []struct {
		name string
		node *expression.Node
		want int
	}{
		{
			// sqrt(x): 2 nodos
			"sqrt(x)",
			expression.NewUnary(expression.NodeSqrt, expression.NewVariable()),
			2,
		},
		{
			// ln(x+1): NodeLn + NodeAdd + NodeVariable + NodeConstant = 4
			"ln(x+1)",
			expression.NewUnary(expression.NodeLn,
				expression.NewBinary(expression.NodeAdd,
					expression.NewVariable(), expression.NewConstant(1))),
			4,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := expression.Count(tt.node)
			if got != tt.want {
				t.Errorf("got %d, want %d", got, tt.want)
			}
		})
	}
}

func TestClone_UnaryIndependence(t *testing.T) {
	// Clone de sqrt(x): modificar el clon no afecta el original
	original := expression.NewUnary(expression.NodeSqrt, expression.NewVariable())
	clone := expression.Clone(original)

	clone.Left = expression.NewConstant(42)

	if original.Left.Type != expression.NodeVariable {
		t.Error("modificar clone.Left afectó original.Left en nodo unario")
	}
}

// ─── Format — nodos unarios ───────────────────────────────────────────────

func TestFormat_Unary(t *testing.T) {
	tests := []struct {
		name   string
		node   *expression.Node
		expect string
	}{
		{
			"sqrt(x)",
			expression.NewUnary(expression.NodeSqrt, expression.NewVariable()),
			"sqrt(x)",
		},
		{
			"ln(x)",
			expression.NewUnary(expression.NodeLn, expression.NewVariable()),
			"ln(x)",
		},
		{
			// El argumento binario se parentesiza por la regla de Format
			"sqrt(x + 2)",
			expression.NewUnary(expression.NodeSqrt,
				expression.NewBinary(expression.NodeAdd,
					expression.NewVariable(), expression.NewConstant(2))),
			"sqrt((x + 2))",
		},
		{
			"ln(x - 1)",
			expression.NewUnary(expression.NodeLn,
				expression.NewBinary(expression.NodeSub,
					expression.NewVariable(), expression.NewConstant(1))),
			"ln((x - 1))",
		},
		{
			// Composición: sqrt dentro de ln está fuera del scope actual,
			// pero la representación debe ser correcta si ocurriera
			"ln(sqrt(x))",
			expression.NewUnary(expression.NodeLn,
				expression.NewUnary(expression.NodeSqrt,
					expression.NewVariable())),
			"ln(sqrt(x))",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := expression.Format(tt.node)
			if got != tt.expect {
				t.Errorf("got %q, want %q", got, tt.expect)
			}
		})
	}
}
