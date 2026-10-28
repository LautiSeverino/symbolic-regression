package expression_test

// format_test.go — Fase 13: tests de Format con precedencia mínima.
//
// Cubre:
//  - Precedencia de Add/Sub (1) vs Mul/Div (2) vs Pow (3)
//  - Hijo izquierdo a misma precedencia: sin paréntesis (excepto Pow)
//  - Hijo derecho a misma precedencia: con paréntesis (excepto Pow derecho-derecho)
//  - NodePow: asociatividad a derecha, base negativa, composiciones
//  - Funciones unarias: un solo par de paréntesis del f(·)
//  - Expresiones representativas del spec

import (
	"testing"

	"github.com/LautiSeverino/symbolic-regression/internal/expression"
)

// ─── Add/Sub (prec 1) ──────────────────────────────────────────────────────

func TestFormat_AddPrec_MulLeft(t *testing.T) {
	// Add(Mul(2,x), 1) → "2 * x + 1"
	tree := expression.NewBinary(expression.NodeAdd,
		expression.NewBinary(expression.NodeMul,
			expression.NewConstant(2), expression.NewVariable()),
		expression.NewConstant(1))
	if got := expression.Format(tree); got != "2 * x + 1" {
		t.Errorf("got %q, want %q", got, "2 * x + 1")
	}
}

func TestFormat_AddPrec_AddLeft(t *testing.T) {
	// Add(Add(x,1), 2) → "x + 1 + 2"  (hijo izq misma prec: sin paréntesis)
	tree := expression.NewBinary(expression.NodeAdd,
		expression.NewBinary(expression.NodeAdd,
			expression.NewVariable(), expression.NewConstant(1)),
		expression.NewConstant(2))
	if got := expression.Format(tree); got != "x + 1 + 2" {
		t.Errorf("got %q, want %q", got, "x + 1 + 2")
	}
}

func TestFormat_AddPrec_AddRight(t *testing.T) {
	// Add(x, Add(1,2)) → "x + (1 + 2)"  (hijo der misma prec: paréntesis)
	tree := expression.NewBinary(expression.NodeAdd,
		expression.NewVariable(),
		expression.NewBinary(expression.NodeAdd,
			expression.NewConstant(1), expression.NewConstant(2)))
	if got := expression.Format(tree); got != "x + (1 + 2)" {
		t.Errorf("got %q, want %q", got, "x + (1 + 2)")
	}
}

func TestFormat_AddPrec_SubRight(t *testing.T) {
	// Add(x, Sub(1,2)) → "x + (1 - 2)"  (Sub misma prec que Add, der → paréntesis)
	tree := expression.NewBinary(expression.NodeAdd,
		expression.NewVariable(),
		expression.NewBinary(expression.NodeSub,
			expression.NewConstant(1), expression.NewConstant(2)))
	if got := expression.Format(tree); got != "x + (1 - 2)" {
		t.Errorf("got %q, want %q", got, "x + (1 - 2)")
	}
}

func TestFormat_AddPrec_MulRight(t *testing.T) {
	// Add(x, Mul(2,3)) → "x + 2 * 3"  (Mul prec mayor: sin paréntesis)
	tree := expression.NewBinary(expression.NodeAdd,
		expression.NewVariable(),
		expression.NewBinary(expression.NodeMul,
			expression.NewConstant(2), expression.NewConstant(3)))
	if got := expression.Format(tree); got != "x + 2 * 3" {
		t.Errorf("got %q, want %q", got, "x + 2 * 3")
	}
}

func TestFormat_SubPrec_SubLeft(t *testing.T) {
	// Sub(Sub(x,1), 2) → "x - 1 - 2"  (hijo izq misma prec: sin paréntesis)
	tree := expression.NewBinary(expression.NodeSub,
		expression.NewBinary(expression.NodeSub,
			expression.NewVariable(), expression.NewConstant(1)),
		expression.NewConstant(2))
	if got := expression.Format(tree); got != "x - 1 - 2" {
		t.Errorf("got %q, want %q", got, "x - 1 - 2")
	}
}

func TestFormat_SubPrec_SubRight(t *testing.T) {
	// Sub(x, Sub(1,2)) → "x - (1 - 2)"  (hijo der misma prec: paréntesis)
	tree := expression.NewBinary(expression.NodeSub,
		expression.NewVariable(),
		expression.NewBinary(expression.NodeSub,
			expression.NewConstant(1), expression.NewConstant(2)))
	if got := expression.Format(tree); got != "x - (1 - 2)" {
		t.Errorf("got %q, want %q", got, "x - (1 - 2)")
	}
}

func TestFormat_SubPrec_AddRight(t *testing.T) {
	// Sub(x, Add(1,2)) → "x - (1 + 2)"  (Add misma prec que Sub, der → paréntesis)
	tree := expression.NewBinary(expression.NodeSub,
		expression.NewVariable(),
		expression.NewBinary(expression.NodeAdd,
			expression.NewConstant(1), expression.NewConstant(2)))
	if got := expression.Format(tree); got != "x - (1 + 2)" {
		t.Errorf("got %q, want %q", got, "x - (1 + 2)")
	}
}

// ─── Mul/Div (prec 2) ──────────────────────────────────────────────────────

func TestFormat_MulPrec_AddLeft(t *testing.T) {
	// Mul(Add(x,1), 2) → "(x + 1) * 2"  (Add menor prec: paréntesis)
	tree := expression.NewBinary(expression.NodeMul,
		expression.NewBinary(expression.NodeAdd,
			expression.NewVariable(), expression.NewConstant(1)),
		expression.NewConstant(2))
	if got := expression.Format(tree); got != "(x + 1) * 2" {
		t.Errorf("got %q, want %q", got, "(x + 1) * 2")
	}
}

func TestFormat_MulPrec_AddRight(t *testing.T) {
	// Mul(x, Add(2,3)) → "x * (2 + 3)"  (Add menor prec: paréntesis)
	tree := expression.NewBinary(expression.NodeMul,
		expression.NewVariable(),
		expression.NewBinary(expression.NodeAdd,
			expression.NewConstant(2), expression.NewConstant(3)))
	if got := expression.Format(tree); got != "x * (2 + 3)" {
		t.Errorf("got %q, want %q", got, "x * (2 + 3)")
	}
}

func TestFormat_MulPrec_MulLeft(t *testing.T) {
	// Mul(Mul(x,2), 3) → "x * 2 * 3"  (hijo izq misma prec: sin paréntesis)
	tree := expression.NewBinary(expression.NodeMul,
		expression.NewBinary(expression.NodeMul,
			expression.NewVariable(), expression.NewConstant(2)),
		expression.NewConstant(3))
	if got := expression.Format(tree); got != "x * 2 * 3" {
		t.Errorf("got %q, want %q", got, "x * 2 * 3")
	}
}

func TestFormat_MulPrec_MulRight(t *testing.T) {
	// Mul(x, Mul(2,3)) → "x * (2 * 3)"  (hijo der misma prec: paréntesis)
	tree := expression.NewBinary(expression.NodeMul,
		expression.NewVariable(),
		expression.NewBinary(expression.NodeMul,
			expression.NewConstant(2), expression.NewConstant(3)))
	if got := expression.Format(tree); got != "x * (2 * 3)" {
		t.Errorf("got %q, want %q", got, "x * (2 * 3)")
	}
}

func TestFormat_MulPrec_DivRight(t *testing.T) {
	// Mul(x, Div(2,3)) → "x * (2 / 3)"  (Div misma prec que Mul, der → paréntesis)
	tree := expression.NewBinary(expression.NodeMul,
		expression.NewVariable(),
		expression.NewBinary(expression.NodeDiv,
			expression.NewConstant(2), expression.NewConstant(3)))
	if got := expression.Format(tree); got != "x * (2 / 3)" {
		t.Errorf("got %q, want %q", got, "x * (2 / 3)")
	}
}

func TestFormat_DivPrec_BothLowerPrec(t *testing.T) {
	// Div(Add(x,1), Sub(x,2)) → "(x + 1) / (x - 2)"
	tree := expression.NewBinary(expression.NodeDiv,
		expression.NewBinary(expression.NodeAdd,
			expression.NewVariable(), expression.NewConstant(1)),
		expression.NewBinary(expression.NodeSub,
			expression.NewVariable(), expression.NewConstant(2)))
	if got := expression.Format(tree); got != "(x + 1) / (x - 2)" {
		t.Errorf("got %q, want %q", got, "(x + 1) / (x - 2)")
	}
}

func TestFormat_DivPrec_DivLeft(t *testing.T) {
	// Div(Div(x,2), 3) → "x / 2 / 3"  (hijo izq misma prec: sin paréntesis)
	tree := expression.NewBinary(expression.NodeDiv,
		expression.NewBinary(expression.NodeDiv,
			expression.NewVariable(), expression.NewConstant(2)),
		expression.NewConstant(3))
	if got := expression.Format(tree); got != "x / 2 / 3" {
		t.Errorf("got %q, want %q", got, "x / 2 / 3")
	}
}

func TestFormat_DivPrec_DivRight(t *testing.T) {
	// Div(x, Div(2,3)) → "x / (2 / 3)"  (hijo der misma prec: paréntesis)
	tree := expression.NewBinary(expression.NodeDiv,
		expression.NewVariable(),
		expression.NewBinary(expression.NodeDiv,
			expression.NewConstant(2), expression.NewConstant(3)))
	if got := expression.Format(tree); got != "x / (2 / 3)" {
		t.Errorf("got %q, want %q", got, "x / (2 / 3)")
	}
}

// ─── NodePow (prec 3) ──────────────────────────────────────────────────────

func TestFormat_Pow_Simple(t *testing.T) {
	// Pow(x, 2) → "x ^ 2"
	tree := expression.NewBinary(expression.NodePow,
		expression.NewVariable(), expression.NewConstant(2))
	if got := expression.Format(tree); got != "x ^ 2" {
		t.Errorf("got %q, want %q", got, "x ^ 2")
	}
}

func TestFormat_Pow_NegExponent(t *testing.T) {
	// Pow(x, -1) → "x ^ -1"
	tree := expression.NewBinary(expression.NodePow,
		expression.NewVariable(), expression.NewConstant(-1))
	if got := expression.Format(tree); got != "x ^ -1" {
		t.Errorf("got %q, want %q", got, "x ^ -1")
	}
}

func TestFormat_Pow_AddInLeft(t *testing.T) {
	// Pow(Add(x,1), 3) → "(x + 1) ^ 3"  (Add prec 1 < Pow prec 3)
	tree := expression.NewBinary(expression.NodePow,
		expression.NewBinary(expression.NodeAdd,
			expression.NewVariable(), expression.NewConstant(1)),
		expression.NewConstant(3))
	if got := expression.Format(tree); got != "(x + 1) ^ 3" {
		t.Errorf("got %q, want %q", got, "(x + 1) ^ 3")
	}
}

func TestFormat_Pow_MulInLeft(t *testing.T) {
	// Pow(Mul(x,2), 3) → "(x * 2) ^ 3"  (Mul prec 2 < Pow prec 3)
	tree := expression.NewBinary(expression.NodePow,
		expression.NewBinary(expression.NodeMul,
			expression.NewVariable(), expression.NewConstant(2)),
		expression.NewConstant(3))
	if got := expression.Format(tree); got != "(x * 2) ^ 3" {
		t.Errorf("got %q, want %q", got, "(x * 2) ^ 3")
	}
}

func TestFormat_Pow_PowInLeft(t *testing.T) {
	// Pow(Pow(x,2), 3) → "(x ^ 2) ^ 3"
	// Hijo izq con misma prec que Pow → paréntesis (evita x^2^3 que por right-assoc sería x^(2^3))
	tree := expression.NewBinary(expression.NodePow,
		expression.NewBinary(expression.NodePow,
			expression.NewVariable(), expression.NewConstant(2)),
		expression.NewConstant(3))
	if got := expression.Format(tree); got != "(x ^ 2) ^ 3" {
		t.Errorf("got %q, want %q", got, "(x ^ 2) ^ 3")
	}
}

func TestFormat_Pow_PowInRight(t *testing.T) {
	// Pow(x, Pow(2,3)) → "x ^ 2 ^ 3"
	// Hijo der con misma prec que Pow → sin paréntesis (asociatividad a derecha)
	tree := expression.NewBinary(expression.NodePow,
		expression.NewVariable(),
		expression.NewBinary(expression.NodePow,
			expression.NewConstant(2), expression.NewConstant(3)))
	if got := expression.Format(tree); got != "x ^ 2 ^ 3" {
		t.Errorf("got %q, want %q", got, "x ^ 2 ^ 3")
	}
}

func TestFormat_Pow_NegConstBase(t *testing.T) {
	// Pow(Const(-3), 2) → "(-3) ^ 2"
	// Constante negativa como base: paréntesis para evitar ambigüedad con -(3^2)
	tree := expression.NewBinary(expression.NodePow,
		expression.NewConstant(-3), expression.NewConstant(2))
	if got := expression.Format(tree); got != "(-3) ^ 2" {
		t.Errorf("got %q, want %q", got, "(-3) ^ 2")
	}
}

func TestFormat_Pow_PosConstBase(t *testing.T) {
	// Pow(Const(3), 2) → "3 ^ 2"  (positiva: sin paréntesis)
	tree := expression.NewBinary(expression.NodePow,
		expression.NewConstant(3), expression.NewConstant(2))
	if got := expression.Format(tree); got != "3 ^ 2" {
		t.Errorf("got %q, want %q", got, "3 ^ 2")
	}
}

func TestFormat_Pow_InAdd(t *testing.T) {
	// Add(Pow(x,2), x) → "x ^ 2 + x"  (Pow mayor prec: sin paréntesis)
	tree := expression.NewBinary(expression.NodeAdd,
		expression.NewBinary(expression.NodePow,
			expression.NewVariable(), expression.NewConstant(2)),
		expression.NewVariable())
	if got := expression.Format(tree); got != "x ^ 2 + x" {
		t.Errorf("got %q, want %q", got, "x ^ 2 + x")
	}
}

func TestFormat_Pow_InMul(t *testing.T) {
	// Mul(Pow(x,2), 3) → "x ^ 2 * 3"  (Pow mayor prec que Mul: sin paréntesis)
	tree := expression.NewBinary(expression.NodeMul,
		expression.NewBinary(expression.NodePow,
			expression.NewVariable(), expression.NewConstant(2)),
		expression.NewConstant(3))
	if got := expression.Format(tree); got != "x ^ 2 * 3" {
		t.Errorf("got %q, want %q", got, "x ^ 2 * 3")
	}
}

func TestFormat_Pow_AddRight(t *testing.T) {
	// Pow(x, Add(2,3)) → "x ^ (2 + 3)"  (Add menor prec que Pow: paréntesis)
	tree := expression.NewBinary(expression.NodePow,
		expression.NewVariable(),
		expression.NewBinary(expression.NodeAdd,
			expression.NewConstant(2), expression.NewConstant(3)))
	if got := expression.Format(tree); got != "x ^ (2 + 3)" {
		t.Errorf("got %q, want %q", got, "x ^ (2 + 3)")
	}
}

// ─── Funciones unarias ────────────────────────────────────────────────────

func TestFormat_Sqrt_Atom(t *testing.T) {
	got := expression.Format(expression.NewUnary(expression.NodeSqrt, expression.NewVariable()))
	if got != "sqrt(x)" {
		t.Errorf("got %q, want %q", got, "sqrt(x)")
	}
}

func TestFormat_Sqrt_Add(t *testing.T) {
	// sqrt(x + 2) — los paréntesis de la función agrupan; no se añaden extra
	tree := expression.NewUnary(expression.NodeSqrt,
		expression.NewBinary(expression.NodeAdd,
			expression.NewVariable(), expression.NewConstant(2)))
	if got := expression.Format(tree); got != "sqrt(x + 2)" {
		t.Errorf("got %q, want %q", got, "sqrt(x + 2)")
	}
}

func TestFormat_Ln_Mul(t *testing.T) {
	tree := expression.NewUnary(expression.NodeLn,
		expression.NewBinary(expression.NodeMul,
			expression.NewVariable(), expression.NewConstant(2)))
	if got := expression.Format(tree); got != "ln(x * 2)" {
		t.Errorf("got %q, want %q", got, "ln(x * 2)")
	}
}

func TestFormat_Abs_Sub(t *testing.T) {
	tree := expression.NewUnary(expression.NodeAbs,
		expression.NewBinary(expression.NodeSub,
			expression.NewVariable(), expression.NewConstant(3)))
	if got := expression.Format(tree); got != "abs(x - 3)" {
		t.Errorf("got %q, want %q", got, "abs(x - 3)")
	}
}

func TestFormat_Exp_Add(t *testing.T) {
	tree := expression.NewUnary(expression.NodeExp,
		expression.NewBinary(expression.NodeAdd,
			expression.NewVariable(), expression.NewConstant(1)))
	if got := expression.Format(tree); got != "exp(x + 1)" {
		t.Errorf("got %q, want %q", got, "exp(x + 1)")
	}
}

func TestFormat_Sin_Mul(t *testing.T) {
	tree := expression.NewUnary(expression.NodeSin,
		expression.NewBinary(expression.NodeMul,
			expression.NewVariable(), expression.NewConstant(2)))
	if got := expression.Format(tree); got != "sin(x * 2)" {
		t.Errorf("got %q, want %q", got, "sin(x * 2)")
	}
}

func TestFormat_Cos_Pow(t *testing.T) {
	tree := expression.NewUnary(expression.NodeCos,
		expression.NewBinary(expression.NodePow,
			expression.NewVariable(), expression.NewConstant(2)))
	if got := expression.Format(tree); got != "cos(x ^ 2)" {
		t.Errorf("got %q, want %q", got, "cos(x ^ 2)")
	}
}

func TestFormat_NestedUnary(t *testing.T) {
	tree := expression.NewUnary(expression.NodeSin,
		expression.NewUnary(expression.NodeCos, expression.NewVariable()))
	if got := expression.Format(tree); got != "sin(cos(x))" {
		t.Errorf("got %q, want %q", got, "sin(cos(x))")
	}
}

func TestFormat_Ln_Sqrt(t *testing.T) {
	tree := expression.NewUnary(expression.NodeLn,
		expression.NewUnary(expression.NodeSqrt, expression.NewVariable()))
	if got := expression.Format(tree); got != "ln(sqrt(x))" {
		t.Errorf("got %q, want %q", got, "ln(sqrt(x))")
	}
}

// ─── Expresiones representativas del spec ─────────────────────────────────

func TestFormat_Spec_Linear2x1(t *testing.T) {
	// Add(Mul(2,x), 1) → "2 * x + 1"
	tree := expression.NewBinary(expression.NodeAdd,
		expression.NewBinary(expression.NodeMul,
			expression.NewConstant(2), expression.NewVariable()),
		expression.NewConstant(1))
	if got := expression.Format(tree); got != "2 * x + 1" {
		t.Errorf("got %q, want %q", got, "2 * x + 1")
	}
}

func TestFormat_Spec_Rational(t *testing.T) {
	// (x+1)/(x-2) → "(x + 1) / (x - 2)"
	tree := expression.NewBinary(expression.NodeDiv,
		expression.NewBinary(expression.NodeAdd,
			expression.NewVariable(), expression.NewConstant(1)),
		expression.NewBinary(expression.NodeSub,
			expression.NewVariable(), expression.NewConstant(2)))
	if got := expression.Format(tree); got != "(x + 1) / (x - 2)" {
		t.Errorf("got %q, want %q", got, "(x + 1) / (x - 2)")
	}
}

func TestFormat_Spec_SinOverX(t *testing.T) {
	// sin(x)/x → "sin(x) / x"
	tree := expression.NewBinary(expression.NodeDiv,
		expression.NewUnary(expression.NodeSin, expression.NewVariable()),
		expression.NewVariable())
	if got := expression.Format(tree); got != "sin(x) / x" {
		t.Errorf("got %q, want %q", got, "sin(x) / x")
	}
}

func TestFormat_Spec_Quadratic(t *testing.T) {
	// Add(Add(Pow(x,2), Mul(3,x)), 2) → "x ^ 2 + 3 * x + 2"
	tree := expression.NewBinary(expression.NodeAdd,
		expression.NewBinary(expression.NodeAdd,
			expression.NewBinary(expression.NodePow,
				expression.NewVariable(), expression.NewConstant(2)),
			expression.NewBinary(expression.NodeMul,
				expression.NewConstant(3), expression.NewVariable())),
		expression.NewConstant(2))
	if got := expression.Format(tree); got != "x ^ 2 + 3 * x + 2" {
		t.Errorf("got %q, want %q", got, "x ^ 2 + 3 * x + 2")
	}
}

func TestFormat_Spec_SqrtX2Plus1(t *testing.T) {
	// sqrt(x^2 + 1) → "sqrt(x ^ 2 + 1)"
	tree := expression.NewUnary(expression.NodeSqrt,
		expression.NewBinary(expression.NodeAdd,
			expression.NewBinary(expression.NodePow,
				expression.NewVariable(), expression.NewConstant(2)),
			expression.NewConstant(1)))
	if got := expression.Format(tree); got != "sqrt(x ^ 2 + 1)" {
		t.Errorf("got %q, want %q", got, "sqrt(x ^ 2 + 1)")
	}
}

func TestFormat_Spec_ExpPlusCos(t *testing.T) {
	// exp(x) + cos(x) → "exp(x) + cos(x)"
	tree := expression.NewBinary(expression.NodeAdd,
		expression.NewUnary(expression.NodeExp, expression.NewVariable()),
		expression.NewUnary(expression.NodeCos, expression.NewVariable()))
	if got := expression.Format(tree); got != "exp(x) + cos(x)" {
		t.Errorf("got %q, want %q", got, "exp(x) + cos(x)")
	}
}
