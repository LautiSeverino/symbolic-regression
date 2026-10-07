package expression_test

// robustness_test.go — Fase 9: batería exhaustiva de robustez numérica.
//
// Este archivo cubre casos que no están en expression_test.go ni phase8_test.go:
//
//  1. NodeConstant no finito (NaN, ±Inf) → EvalError   [nueva conducta introducida en Fase 9]
//  2. División: overflow a ±Inf por divisor subnormal
//  3. sqrt: dominio extendido (MaxFloat64, valores muy pequeños)
//  4. ln: dominio extendido (SmallestNonzeroFloat64, MaxFloat64)
//  5. NodePow: política de 0^0=1, base unitaria, base negativa con exp entero
//  6. NodeExp: underflow a 0 (resultado válido, no EvalError)
//  7. NodeAbs: valores extremos y propagación de constante no finita
//  8. sin/cos: argumentos extremamente grandes (siguen acotados en [-1,1])
//  9. Expresiones anidadas: sqrt(ln(x)), exp(1/x), sin(1/(x-x)), abs(sqrt(-1)), etc.
// 10. Propagación de EvalError de NodeConstant no finito a través de todos los operadores
// 11. Barrido no-panic: matriz completa de operadores × rango de x
//
// Políticas numéricas del proyecto (documentadas y testeadas aquí):
//   - 0^0    = 1   (convención Go / math.Pow(0,0) = 1)
//   - exp(underflow) = 0   (float64 denormalizado a 0; finito, no error)
//   - 1/SmallestNonzeroFloat64 → overflow a +Inf → EvalError
//   - NodeConstant(NaN) y NodeConstant(±Inf) → EvalError inmediato

import (
	"math"
	"testing"

	"github.com/LautiSeverino/symbolic-regression/internal/expression"
)

// ─── Helpers locales ──────────────────────────────────────────────────────

// pow9 construye base^exp con constantes.
func pow9(base, exp float64) *expression.Node {
	return expression.NewBinary(expression.NodePow,
		expression.NewConstant(base), expression.NewConstant(exp))
}

// div9 construye num/den con constantes.
func div9(num, den float64) *expression.Node {
	return expression.NewBinary(expression.NodeDiv,
		expression.NewConstant(num), expression.NewConstant(den))
}

// unary9 construye op(arg) donde arg es una constante.
func unary9(op expression.NodeType, arg float64) *expression.Node {
	return expression.NewUnary(op, expression.NewConstant(arg))
}

// ══════════════════════════════════════════════════════════════════════════
// 1. NodeConstant no finito — nueva conducta de Fase 9
// ══════════════════════════════════════════════════════════════════════════

func TestEvaluate_NonFiniteConstant_DirectError(t *testing.T) {
	// Una constante con valor NaN o ±Inf debe retornar EvalError inmediatamente.
	// Sin este guard, el valor no finito se propagaría silenciosamente por el árbol.
	cases := []struct {
		name string
		v    float64
	}{
		{"NaN", math.NaN()},
		{"+Inf", math.Inf(1)},
		{"-Inf", math.Inf(-1)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := expression.Evaluate(expression.NewConstant(tc.v), 0)
			if err == nil {
				t.Errorf("NodeConstant(%v): se esperaba EvalError, se obtuvo nil", tc.v)
			}
		})
	}
}

func TestEvaluate_NonFiniteConstant_FiniteConstantsUnaffected(t *testing.T) {
	// Los valores finitos, incluyendo cero y los extremos de float64, deben seguir funcionando.
	cases := []float64{
		0, 1, -1, 3.14, -2.71,
		math.MaxFloat64, -math.MaxFloat64,
		math.SmallestNonzeroFloat64,
	}
	for _, v := range cases {
		got, err := expression.Evaluate(expression.NewConstant(v), 0)
		if err != nil {
			t.Errorf("NodeConstant(%v): error inesperado: %v", v, err)
			continue
		}
		if got != v {
			t.Errorf("NodeConstant(%v): got %v", v, got)
		}
	}
}

// ══════════════════════════════════════════════════════════════════════════
// 2. División — casos extremos
// ══════════════════════════════════════════════════════════════════════════

func TestEvaluate_Div_Extreme(t *testing.T) {
	t.Run("1/SmallestNonzero overflow→EvalError", func(t *testing.T) {
		// 1 / 5e-324 ≈ 2^1074 >> MaxFloat64 → +Inf → EvalError
		node := div9(1, math.SmallestNonzeroFloat64)
		_, err := expression.Evaluate(node, 0)
		if err == nil {
			t.Error("1/SmallestNonzero: se esperaba EvalError por overflow, se obtuvo nil")
		}
	})

	t.Run("MaxFloat64/2 válido", func(t *testing.T) {
		node := div9(math.MaxFloat64, 2)
		got, err := expression.Evaluate(node, 0)
		if err != nil {
			t.Fatalf("error inesperado: %v", err)
		}
		if got != math.MaxFloat64/2 {
			t.Errorf("MaxFloat64/2: got %v", got)
		}
	})

	t.Run("1/MaxFloat64 válido (resultado muy pequeño)", func(t *testing.T) {
		node := div9(1, math.MaxFloat64)
		got, err := expression.Evaluate(node, 0)
		if err != nil {
			t.Fatalf("error inesperado: %v", err)
		}
		if math.IsNaN(got) || math.IsInf(got, 0) || got < 0 {
			t.Errorf("1/MaxFloat64: resultado no finito: %v", got)
		}
	})

	t.Run("propagación error hijo izquierdo", func(t *testing.T) {
		// (NaN_const) / 2 → error de NodeConstant propagado
		node := expression.NewBinary(expression.NodeDiv,
			expression.NewConstant(math.NaN()), expression.NewConstant(2))
		_, err := expression.Evaluate(node, 0)
		if err == nil {
			t.Error("NaN_const/2: se esperaba EvalError propagado")
		}
	})

	t.Run("propagación error hijo derecho", func(t *testing.T) {
		// 2 / (Inf_const) → error de NodeConstant propagado
		node := expression.NewBinary(expression.NodeDiv,
			expression.NewConstant(2), expression.NewConstant(math.Inf(1)))
		_, err := expression.Evaluate(node, 0)
		if err == nil {
			t.Error("2/Inf_const: se esperaba EvalError propagado")
		}
	})
}

// ══════════════════════════════════════════════════════════════════════════
// 3. sqrt — dominio extendido
// ══════════════════════════════════════════════════════════════════════════

func TestEvaluate_Sqrt_ExtendedDomain(t *testing.T) {
	t.Run("sqrt(MaxFloat64) válido ≈ 1.34e154", func(t *testing.T) {
		got, err := expression.Evaluate(unary9(expression.NodeSqrt, math.MaxFloat64), 0)
		if err != nil {
			t.Fatalf("error inesperado: %v", err)
		}
		if math.IsNaN(got) || math.IsInf(got, 0) || got <= 0 {
			t.Errorf("sqrt(MaxFloat64) = %v: se esperaba finito positivo", got)
		}
		// Verificación de consistencia: resultado^2 ≈ MaxFloat64
		if got*got > math.MaxFloat64 {
			t.Errorf("sqrt(MaxFloat64)^2 overflowea: sqrt=%v", got)
		}
	})

	t.Run("sqrt(1e-300) válido (raíz de subnormal grande)", func(t *testing.T) {
		got, err := expression.Evaluate(unary9(expression.NodeSqrt, 1e-300), 0)
		if err != nil {
			t.Fatalf("error inesperado: %v", err)
		}
		if got <= 0 || math.IsInf(got, 0) || math.IsNaN(got) {
			t.Errorf("sqrt(1e-300) = %v: se esperaba positivo finito", got)
		}
	})

	t.Run("sqrt(SmallestNonzeroFloat64) válido", func(t *testing.T) {
		got, err := expression.Evaluate(unary9(expression.NodeSqrt, math.SmallestNonzeroFloat64), 0)
		if err != nil {
			t.Fatalf("error inesperado: %v", err)
		}
		if got < 0 || math.IsNaN(got) {
			t.Errorf("sqrt(SmallestNonzero) = %v: se esperaba ≥ 0", got)
		}
	})
}

// ══════════════════════════════════════════════════════════════════════════
// 4. ln — dominio extendido
// ══════════════════════════════════════════════════════════════════════════

func TestEvaluate_Ln_ExtendedDomain(t *testing.T) {
	t.Run("ln(SmallestNonzeroFloat64) válido ≈ −744", func(t *testing.T) {
		// ln(5e-324) ≈ -744: finito negativo, dominio positivo → no error
		got, err := expression.Evaluate(unary9(expression.NodeLn, math.SmallestNonzeroFloat64), 0)
		if err != nil {
			t.Fatalf("error inesperado: %v", err)
		}
		if math.IsNaN(got) || math.IsInf(got, 0) {
			t.Errorf("ln(SmallestNonzero) = %v: se esperaba finito", got)
		}
		if got >= 0 {
			t.Errorf("ln(SmallestNonzero) = %v: se esperaba negativo", got)
		}
	})

	t.Run("ln(MaxFloat64) válido ≈ 709", func(t *testing.T) {
		got, err := expression.Evaluate(unary9(expression.NodeLn, math.MaxFloat64), 0)
		if err != nil {
			t.Fatalf("error inesperado: %v", err)
		}
		if math.IsNaN(got) || math.IsInf(got, 0) {
			t.Errorf("ln(MaxFloat64) = %v: se esperaba finito", got)
		}
		if got <= 0 {
			t.Errorf("ln(MaxFloat64) = %v: se esperaba positivo", got)
		}
	})

	t.Run("ln(1) = 0 exacto", func(t *testing.T) {
		got, err := expression.Evaluate(unary9(expression.NodeLn, 1), 0)
		if err != nil {
			t.Fatalf("error inesperado: %v", err)
		}
		if got != 0 {
			t.Errorf("ln(1) = %v, want 0", got)
		}
	})
}

// ══════════════════════════════════════════════════════════════════════════
// 5. NodePow — política de 0^0 y casos de borde adicionales
// ══════════════════════════════════════════════════════════════════════════

// TestEvaluate_Pow_ZeroZero documenta y verifica la política adoptada: 0^0 = 1.
// Fundamento: Go sigue la convención IEEE 754 / C99: math.Pow(0, 0) = 1.
// Esta elección es consistente con la mayoría de lenguajes de programación.
func TestEvaluate_Pow_ZeroZero(t *testing.T) {
	got, err := expression.Evaluate(pow9(0, 0), 0)
	if err != nil {
		t.Fatalf("0^0: error inesperado (política del proyecto = 1): %v", err)
	}
	if got != 1 {
		t.Errorf("0^0 = %v, want 1 (política del proyecto)", got)
	}
}

func TestEvaluate_Pow_ZeroBase(t *testing.T) {
	cases := []struct {
		exp  float64
		want float64
	}{
		{1, 0}, // 0^1 = 0
		{2, 0}, // 0^2 = 0
		{4, 0}, // 0^4 = 0
	}
	for _, tc := range cases {
		got, err := expression.Evaluate(pow9(0, tc.exp), 0)
		if err != nil {
			t.Errorf("0^%v: error inesperado: %v", tc.exp, err)
			continue
		}
		if got != tc.want {
			t.Errorf("0^%v = %v, want %v", tc.exp, got, tc.want)
		}
	}
}

func TestEvaluate_Pow_UnitBase(t *testing.T) {
	// 1^n = 1 para cualquier exponente finito (incluyendo negativos y fraccionarios).
	exps := []float64{-4, -1, -0.5, 0, 0.5, 1, 2, 4, 1000}
	for _, e := range exps {
		got, err := expression.Evaluate(pow9(1, e), 0)
		if err != nil {
			t.Errorf("1^%v: error inesperado: %v", e, err)
			continue
		}
		if got != 1 {
			t.Errorf("1^%v = %v, want 1", e, got)
		}
	}
}

func TestEvaluate_Pow_NegativeBaseIntegerExp(t *testing.T) {
	// Base negativa con exponente entero: resultado real, no NaN.
	cases := []struct {
		base, exp, want float64
	}{
		{-2, 2, 4},
		{-2, 3, -8},
		{-3, 4, 81},
		{-2, -1, -0.5},
		{-1, 0, 1}, // (-1)^0 = 1
	}
	for _, tc := range cases {
		got, err := expression.Evaluate(pow9(tc.base, tc.exp), 0)
		if err != nil {
			t.Errorf("(%v)^%v: error inesperado: %v", tc.base, tc.exp, err)
			continue
		}
		if math.Abs(got-tc.want) > 1e-10 {
			t.Errorf("(%v)^%v = %v, want %v", tc.base, tc.exp, got, tc.want)
		}
	}
}

func TestEvaluate_Pow_ViaVariable(t *testing.T) {
	// x^2, x^3, x^-1 vía variable para distintos x
	cases := []struct {
		exp float64
		x   float64
		want float64
	}{
		{2, 3, 9},
		{3, 2, 8},
		{-1, 4, 0.25},
		{0, 99, 1},   // x^0 = 1 para cualquier x finito ≠ 0
		{2, -5, 25},
	}
	for _, tc := range cases {
		node := expression.NewBinary(expression.NodePow,
			expression.NewVariable(), expression.NewConstant(tc.exp))
		got, err := expression.Evaluate(node, tc.x)
		if err != nil {
			t.Errorf("x^%v en x=%v: error inesperado: %v", tc.exp, tc.x, err)
			continue
		}
		if math.Abs(got-tc.want) > 1e-10 {
			t.Errorf("x^%v en x=%v: got %v, want %v", tc.exp, tc.x, got, tc.want)
		}
	}
}

// ══════════════════════════════════════════════════════════════════════════
// 6. NodeExp — underflow a cero (resultado válido, no EvalError)
// ══════════════════════════════════════════════════════════════════════════

// TestEvaluate_Exp_Underflow documenta y verifica la política adoptada:
// exp(−∞ approx) produce 0.0 por underflow; eso es un float64 finito válido,
// no una condición de error. Solo el overflow (+Inf) produce EvalError.
func TestEvaluate_Exp_Underflow(t *testing.T) {
	t.Run("exp(-1000) underflow=0 válido", func(t *testing.T) {
		got, err := expression.Evaluate(unary9(expression.NodeExp, -1000), 0)
		if err != nil {
			t.Fatalf("exp(-1000): error inesperado (underflow debe ser resultado válido 0): %v", err)
		}
		// En float64, exp(-1000) underflowea exactamente a 0.
		if got != 0 {
			t.Errorf("exp(-1000) = %v, se esperaba 0 (underflow a cero)", got)
		}
	})

	t.Run("exp(-500) resultado subnormal pero finito", func(t *testing.T) {
		got, err := expression.Evaluate(unary9(expression.NodeExp, -500), 0)
		if err != nil {
			t.Fatalf("exp(-500): error inesperado: %v", err)
		}
		if got < 0 || math.IsInf(got, 0) || math.IsNaN(got) {
			t.Errorf("exp(-500) = %v: se esperaba finito no-negativo", got)
		}
	})

	t.Run("exp(-100) finito", func(t *testing.T) {
		got, err := expression.Evaluate(unary9(expression.NodeExp, -100), 0)
		if err != nil {
			t.Fatalf("exp(-100): error inesperado: %v", err)
		}
		if got <= 0 || math.IsNaN(got) || math.IsInf(got, 0) {
			t.Errorf("exp(-100) = %v: se esperaba positivo finito", got)
		}
	})
}

// ══════════════════════════════════════════════════════════════════════════
// 7. NodeAbs — valores extremos y constantes no finitas
// ══════════════════════════════════════════════════════════════════════════

func TestEvaluate_Abs_ExtremValues(t *testing.T) {
	cases := []struct {
		name string
		arg  float64
		want float64
	}{
		{"MaxFloat64", math.MaxFloat64, math.MaxFloat64},
		{"-MaxFloat64", -math.MaxFloat64, math.MaxFloat64},
		{"SmallestNonzero", math.SmallestNonzeroFloat64, math.SmallestNonzeroFloat64},
		{"-SmallestNonzero", -math.SmallestNonzeroFloat64, math.SmallestNonzeroFloat64},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := expression.Evaluate(unary9(expression.NodeAbs, tc.arg), 0)
			if err != nil {
				t.Fatalf("abs(%v): error inesperado: %v", tc.arg, err)
			}
			if got != tc.want {
				t.Errorf("abs(%v) = %v, want %v", tc.arg, got, tc.want)
			}
		})
	}
}

func TestEvaluate_Abs_NonFiniteConstantPropagates(t *testing.T) {
	// Con el guard en NodeConstant, abs(NaN) y abs(±Inf) devuelven EvalError
	// porque NodeConstant ya falla antes de que la operación abs se ejecute.
	cases := []struct {
		name string
		arg  float64
	}{
		{"abs(NaN_const)", math.NaN()},
		{"abs(+Inf_const)", math.Inf(1)},
		{"abs(-Inf_const)", math.Inf(-1)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			node := expression.NewUnary(expression.NodeAbs, expression.NewConstant(tc.arg))
			_, err := expression.Evaluate(node, 0)
			if err == nil {
				t.Errorf("abs(%v): se esperaba EvalError, se obtuvo nil (valor no finito silencioso)", tc.arg)
			}
		})
	}
}

// ══════════════════════════════════════════════════════════════════════════
// 8. sin/cos — argumentos extremamente grandes
// ══════════════════════════════════════════════════════════════════════════

func TestEvaluate_Sin_ExtremeArgsBounded(t *testing.T) {
	// sin devuelve valores en [-1,1] para cualquier argumento finito, sin importar la magnitud.
	node := expression.NewUnary(expression.NodeSin, expression.NewVariable())
	extremes := []float64{1e6, 1e12, 1e15, -1e12, -1e15, math.MaxFloat64 / 2}
	for _, x := range extremes {
		got, err := expression.Evaluate(node, x)
		if err != nil {
			t.Errorf("sin(%v): error inesperado: %v", x, err)
			continue
		}
		if got < -1-1e-12 || got > 1+1e-12 {
			t.Errorf("sin(%v) = %v: fuera de [-1,1]", x, got)
		}
	}
}

func TestEvaluate_Cos_ExtremeArgsBounded(t *testing.T) {
	node := expression.NewUnary(expression.NodeCos, expression.NewVariable())
	extremes := []float64{1e6, 1e12, 1e15, -1e12, -1e15, math.MaxFloat64 / 2}
	for _, x := range extremes {
		got, err := expression.Evaluate(node, x)
		if err != nil {
			t.Errorf("cos(%v): error inesperado: %v", x, err)
			continue
		}
		if got < -1-1e-12 || got > 1+1e-12 {
			t.Errorf("cos(%v) = %v: fuera de [-1,1]", x, got)
		}
	}
}

func TestEvaluate_SinCos_NonFiniteConstantPropagates(t *testing.T) {
	cases := []struct {
		name string
		node *expression.Node
	}{
		{"sin(NaN_const)", expression.NewUnary(expression.NodeSin, expression.NewConstant(math.NaN()))},
		{"sin(+Inf_const)", expression.NewUnary(expression.NodeSin, expression.NewConstant(math.Inf(1)))},
		{"cos(NaN_const)", expression.NewUnary(expression.NodeCos, expression.NewConstant(math.NaN()))},
		{"cos(-Inf_const)", expression.NewUnary(expression.NodeCos, expression.NewConstant(math.Inf(-1)))},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := expression.Evaluate(tc.node, 0)
			if err == nil {
				t.Errorf("%s: se esperaba EvalError por constante no finita", tc.name)
			}
		})
	}
}

// ══════════════════════════════════════════════════════════════════════════
// 9. Expresiones anidadas
// ══════════════════════════════════════════════════════════════════════════

func TestEvaluate_Nested_SqrtLn(t *testing.T) {
	// sqrt(ln(x)):
	//   x > 1  → ln(x) > 0, sqrt válido
	//   0 < x < 1 → ln(x) < 0, sqrt de negativo → EvalError
	//   x ≤ 0  → ln inválido → EvalError
	node := expression.NewUnary(expression.NodeSqrt,
		expression.NewUnary(expression.NodeLn, expression.NewVariable()))

	t.Run("sqrt(ln(e))=1 válido", func(t *testing.T) {
		got, err := expression.Evaluate(node, math.E)
		if err != nil {
			t.Fatalf("x=e: error inesperado: %v", err)
		}
		if math.Abs(got-1) > 1e-10 {
			t.Errorf("sqrt(ln(e)) = %v, want 1", got)
		}
	})

	t.Run("sqrt(ln(0.5)) EvalError (raíz de negativo)", func(t *testing.T) {
		_, err := expression.Evaluate(node, 0.5)
		if err == nil {
			t.Error("sqrt(ln(0.5)): se esperaba EvalError")
		}
	})

	t.Run("sqrt(ln(0)) EvalError (ln no positivo)", func(t *testing.T) {
		_, err := expression.Evaluate(node, 0)
		if err == nil {
			t.Error("sqrt(ln(0)): se esperaba EvalError")
		}
	})

	t.Run("sqrt(ln(-1)) EvalError (ln de negativo)", func(t *testing.T) {
		_, err := expression.Evaluate(node, -1)
		if err == nil {
			t.Error("sqrt(ln(-1)): se esperaba EvalError")
		}
	})
}

func TestEvaluate_Nested_ExpDiv(t *testing.T) {
	// exp(1/x): para x≠0 válido; para x=0 → EvalError por 1/0
	node := expression.NewUnary(expression.NodeExp,
		expression.NewBinary(expression.NodeDiv,
			expression.NewConstant(1),
			expression.NewVariable()))

	t.Run("exp(1/1)=e válido", func(t *testing.T) {
		got, err := expression.Evaluate(node, 1)
		if err != nil {
			t.Fatalf("x=1: error inesperado: %v", err)
		}
		if math.Abs(got-math.E) > 1e-10 {
			t.Errorf("exp(1/1) = %v, want %v", got, math.E)
		}
	})

	t.Run("exp(1/0) EvalError (divisor cero)", func(t *testing.T) {
		_, err := expression.Evaluate(node, 0)
		if err == nil {
			t.Error("exp(1/0): se esperaba EvalError")
		}
	})

	t.Run("exp(1/(-1))=1/e válido", func(t *testing.T) {
		got, err := expression.Evaluate(node, -1)
		if err != nil {
			t.Fatalf("x=-1: error inesperado: %v", err)
		}
		if math.Abs(got-1/math.E) > 1e-10 {
			t.Errorf("exp(1/(-1)) = %v, want %v", got, 1/math.E)
		}
	})
}

func TestEvaluate_Nested_SinDivXminusX(t *testing.T) {
	// sin(1/(x-x)): x-x=0 exacto en float64 para cualquier x → 1/0 → EvalError
	node := expression.NewUnary(expression.NodeSin,
		expression.NewBinary(expression.NodeDiv,
			expression.NewConstant(1),
			expression.NewBinary(expression.NodeSub,
				expression.NewVariable(), expression.NewVariable())))

	for _, x := range []float64{-100, -1, 0, 1, 100} {
		_, err := expression.Evaluate(node, x)
		if err == nil {
			t.Errorf("sin(1/(x-x)) x=%v: se esperaba EvalError (x-x=0 siempre)", x)
		}
	}
}

func TestEvaluate_Nested_AbsSqrtNeg(t *testing.T) {
	// abs(sqrt(-1)): sqrt(-1) → EvalError propagado a abs
	node := expression.NewUnary(expression.NodeAbs,
		expression.NewUnary(expression.NodeSqrt, expression.NewConstant(-1)))
	_, err := expression.Evaluate(node, 0)
	if err == nil {
		t.Error("abs(sqrt(-1)): se esperaba EvalError propagado desde sqrt")
	}
}

func TestEvaluate_Nested_CosExpOverflow(t *testing.T) {
	// cos(exp(1000)): exp(1000)→+Inf → EvalError propagado a cos
	node := expression.NewUnary(expression.NodeCos,
		expression.NewUnary(expression.NodeExp, expression.NewConstant(1000)))
	_, err := expression.Evaluate(node, 0)
	if err == nil {
		t.Error("cos(exp(1000)): se esperaba EvalError propagado desde exp")
	}
}

func TestEvaluate_Nested_SqrtXSquaredPlus1(t *testing.T) {
	// sqrt(x^2 + 1): siempre válido porque x^2+1 ≥ 1 > 0 para cualquier x finito.
	node := expression.NewUnary(expression.NodeSqrt,
		expression.NewBinary(expression.NodeAdd,
			expression.NewBinary(expression.NodePow,
				expression.NewVariable(), expression.NewConstant(2)),
			expression.NewConstant(1)))

	xValues := []float64{-1e6, -100, -1, 0, 1, 100, 1e6}
	for _, x := range xValues {
		got, err := expression.Evaluate(node, x)
		if err != nil {
			t.Errorf("sqrt(x^2+1) x=%v: error inesperado: %v", x, err)
			continue
		}
		want := math.Sqrt(x*x + 1)
		if math.Abs(got-want) > 1e-8 {
			t.Errorf("sqrt(x^2+1) x=%v: got %v, want %v", x, got, want)
		}
	}
}

func TestEvaluate_Nested_LnAbsX(t *testing.T) {
	// ln(abs(x)): para x≠0 siempre tiene abs(x)>0, ln válido; para x=0 ln(0) → EvalError
	node := expression.NewUnary(expression.NodeLn,
		expression.NewUnary(expression.NodeAbs, expression.NewVariable()))

	t.Run("ln(abs(-3))=ln(3) válido", func(t *testing.T) {
		got, err := expression.Evaluate(node, -3)
		if err != nil {
			t.Fatalf("x=-3: error inesperado: %v", err)
		}
		if math.Abs(got-math.Log(3)) > 1e-10 {
			t.Errorf("ln(abs(-3)) = %v, want %v", got, math.Log(3))
		}
	})

	t.Run("ln(abs(-1))=0 válido", func(t *testing.T) {
		got, err := expression.Evaluate(node, -1)
		if err != nil {
			t.Fatalf("x=-1: error inesperado: %v", err)
		}
		if math.Abs(got-0) > 1e-10 {
			t.Errorf("ln(abs(-1)) = %v, want 0", got)
		}
	})

	t.Run("ln(abs(0)) EvalError (ln(0) inválido)", func(t *testing.T) {
		_, err := expression.Evaluate(node, 0)
		if err == nil {
			t.Error("ln(abs(0)): se esperaba EvalError")
		}
	})
}

func TestEvaluate_Nested_AbsSinX(t *testing.T) {
	// abs(sin(x)): siempre válido para cualquier x finito
	node := expression.NewUnary(expression.NodeAbs,
		expression.NewUnary(expression.NodeSin, expression.NewVariable()))

	for _, x := range []float64{-100, -1, 0, 1, 100, 1e9} {
		got, err := expression.Evaluate(node, x)
		if err != nil {
			t.Errorf("abs(sin(%v)): error inesperado: %v", x, err)
			continue
		}
		if got < 0 || got > 1+1e-12 {
			t.Errorf("abs(sin(%v)) = %v: debe estar en [0,1]", x, got)
		}
	}
}

// ══════════════════════════════════════════════════════════════════════════
// 10. Propagación de EvalError de NodeConstant a través de todos los operadores
// ══════════════════════════════════════════════════════════════════════════

func TestEvaluate_NonFiniteConstant_PropagatesThrough(t *testing.T) {
	// Verifica que el EvalError producido por NodeConstant(NaN) se propague
	// correctamente a través de todas las operaciones disponibles.
	// Prueba tanto con NaN como con ±Inf para mayor cobertura.
	type nodeFactory func(bad *expression.Node) *expression.Node
	cases := []struct {
		name    string
		factory nodeFactory
	}{
		{"Add(x, NaN)", func(bad *expression.Node) *expression.Node {
			return expression.NewBinary(expression.NodeAdd, expression.NewVariable(), bad)
		}},
		{"Sub(NaN, x)", func(bad *expression.Node) *expression.Node {
			return expression.NewBinary(expression.NodeSub, bad, expression.NewVariable())
		}},
		{"Mul(x, NaN)", func(bad *expression.Node) *expression.Node {
			return expression.NewBinary(expression.NodeMul, expression.NewVariable(), bad)
		}},
		{"Div(x, NaN)", func(bad *expression.Node) *expression.Node {
			return expression.NewBinary(expression.NodeDiv, expression.NewVariable(), bad)
		}},
		{"Div(NaN, x)", func(bad *expression.Node) *expression.Node {
			return expression.NewBinary(expression.NodeDiv, bad, expression.NewVariable())
		}},
		{"Pow(x, NaN)", func(bad *expression.Node) *expression.Node {
			return expression.NewBinary(expression.NodePow, expression.NewVariable(), bad)
		}},
		{"Pow(NaN, x)", func(bad *expression.Node) *expression.Node {
			return expression.NewBinary(expression.NodePow, bad, expression.NewVariable())
		}},
		{"Sqrt(NaN)", func(bad *expression.Node) *expression.Node {
			return expression.NewUnary(expression.NodeSqrt, bad)
		}},
		{"Ln(NaN)", func(bad *expression.Node) *expression.Node {
			return expression.NewUnary(expression.NodeLn, bad)
		}},
		{"Exp(NaN)", func(bad *expression.Node) *expression.Node {
			return expression.NewUnary(expression.NodeExp, bad)
		}},
		{"Abs(NaN)", func(bad *expression.Node) *expression.Node {
			return expression.NewUnary(expression.NodeAbs, bad)
		}},
		{"Sin(NaN)", func(bad *expression.Node) *expression.Node {
			return expression.NewUnary(expression.NodeSin, bad)
		}},
		{"Cos(Inf)", func(bad *expression.Node) *expression.Node {
			return expression.NewUnary(expression.NodeCos, bad)
		}},
	}

	// Probamos con NaN y con +Inf para cada caso
	badValues := []struct {
		label string
		v     float64
	}{
		{"NaN", math.NaN()},
		{"+Inf", math.Inf(1)},
	}

	for _, bv := range badValues {
		for _, tc := range cases {
			name := tc.name + "/" + bv.label
			t.Run(name, func(t *testing.T) {
				bad := expression.NewConstant(bv.v)
				node := tc.factory(bad)
				_, err := expression.Evaluate(node, 1.0)
				if err == nil {
					t.Errorf("%s: se esperaba EvalError (constante %v no detectada)", name, bv.v)
				}
			})
		}
	}
}

// ══════════════════════════════════════════════════════════════════════════
// 11. Barrido no-panic: todos los operadores × rango amplio de x
// ══════════════════════════════════════════════════════════════════════════

func TestEvaluate_Robustness_NoPanic(t *testing.T) {
	// Verifica que ninguna combinación de operador y valor de x provoque panic.
	// Los errores son esperables (EvalError) y se ignoran; solo los panics fallan.
	xValues := []float64{
		math.SmallestNonzeroFloat64,
		-math.MaxFloat64 / 2, -1e15, -1e6, -100, -3, -1, -0.5, -0.1,
		0,
		0.1, 0.5, 1, 3, 100, 1e6, 1e15, math.MaxFloat64 / 2,
	}

	nodes := []struct {
		name string
		node *expression.Node
	}{
		// Operaciones básicas sobre variable
		{"x+0", expression.NewBinary(expression.NodeAdd, expression.NewVariable(), expression.NewConstant(0))},
		{"x-x", expression.NewBinary(expression.NodeSub, expression.NewVariable(), expression.NewVariable())},
		{"x*2", expression.NewBinary(expression.NodeMul, expression.NewVariable(), expression.NewConstant(2))},
		{"1/x", expression.NewBinary(expression.NodeDiv, expression.NewConstant(1), expression.NewVariable())},
		// Potencias
		{"x^0", expression.NewBinary(expression.NodePow, expression.NewVariable(), expression.NewConstant(0))},
		{"x^2", expression.NewBinary(expression.NodePow, expression.NewVariable(), expression.NewConstant(2))},
		{"x^-1", expression.NewBinary(expression.NodePow, expression.NewVariable(), expression.NewConstant(-1))},
		{"x^4", expression.NewBinary(expression.NodePow, expression.NewVariable(), expression.NewConstant(4))},
		{"0^x", expression.NewBinary(expression.NodePow, expression.NewConstant(0), expression.NewVariable())},
		// Funciones unarias
		{"sqrt(x)", expression.NewUnary(expression.NodeSqrt, expression.NewVariable())},
		{"ln(x)", expression.NewUnary(expression.NodeLn, expression.NewVariable())},
		{"exp(x)", expression.NewUnary(expression.NodeExp, expression.NewVariable())},
		{"abs(x)", expression.NewUnary(expression.NodeAbs, expression.NewVariable())},
		{"sin(x)", expression.NewUnary(expression.NodeSin, expression.NewVariable())},
		{"cos(x)", expression.NewUnary(expression.NodeCos, expression.NewVariable())},
		// Constantes extremas
		{"MaxFloat64 const", expression.NewConstant(math.MaxFloat64)},
		{"-MaxFloat64 const", expression.NewConstant(-math.MaxFloat64)},
		{"SmallestNonzero const", expression.NewConstant(math.SmallestNonzeroFloat64)},
		// Anidadas
		{"sqrt(x^2+1)", expression.NewUnary(expression.NodeSqrt,
			expression.NewBinary(expression.NodeAdd,
				expression.NewBinary(expression.NodePow,
					expression.NewVariable(), expression.NewConstant(2)),
				expression.NewConstant(1)))},
		{"abs(sin(x))", expression.NewUnary(expression.NodeAbs,
			expression.NewUnary(expression.NodeSin, expression.NewVariable()))},
		{"exp(1/x)", expression.NewUnary(expression.NodeExp,
			expression.NewBinary(expression.NodeDiv,
				expression.NewConstant(1), expression.NewVariable()))},
	}

	for _, tc := range nodes {
		for _, x := range xValues {
			tc := tc
			x := x
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Errorf("%s con x=%v: panic inesperado: %v", tc.name, x, r)
					}
				}()
				_, _ = expression.Evaluate(tc.node, x)
			}()
		}
	}
}
