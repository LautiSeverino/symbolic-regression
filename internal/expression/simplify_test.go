package expression_test

// simplify_test.go — Fase 13: tests de Simplify.
//
// Cubre:
//  1. Cada regla de simplificación aprobada (Add/Sub/Mul/Div)
//  2. Casos donde la regla NO debe aplicarse (safe-leaf constraint, divisor-zero)
//  3. Nodos sin regla aplicable
//  4. Nil, hojas, nodos unarios
//  5. Composición y anidamiento de reglas
//  6. Independencia del árbol original
//  7. Idempotencia
//  8. Consistencia de evaluación (Eval(original) == Eval(simplified))
//  9. Preservación de EvalError en casos críticos

import (
	"math"
	"testing"

	"github.com/LautiSeverino/symbolic-regression/internal/expression"
)

// ─── Helper: evaluación consistente ───────────────────────────────────────

// checkEvalConsistent verifica que Simplify(tree) y tree producen el mismo
// resultado de Evaluate (o el mismo tipo de error) para todos los x dados.
func checkEvalConsistent(t *testing.T, name string, tree *expression.Node, xs []float64) {
	t.Helper()
	simplified := expression.Simplify(tree)
	for _, x := range xs {
		origVal, origErr := expression.Evaluate(tree, x)
		simpVal, simpErr := expression.Evaluate(simplified, x)
		origFailed := origErr != nil
		simpFailed := simpErr != nil
		if origFailed != simpFailed {
			t.Errorf("%s x=%v: cambió el comportamiento de error — orig=%v simp=%v",
				name, x, origErr, simpErr)
			continue
		}
		if !origFailed && origVal != simpVal {
			t.Errorf("%s x=%v: valor cambió — orig=%v simp=%v", name, x, origVal, simpVal)
		}
	}
}

var simplifyXRange = []float64{-10, -3, -1, -0.5, 0, 0.5, 1, 3, 10}

// ─── Regla: x + 0 → x ─────────────────────────────────────────────────────

func TestSimplify_AddZeroRight_Var(t *testing.T) {
	tree := expression.NewBinary(expression.NodeAdd,
		expression.NewVariable(), expression.NewConstant(0))
	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeVariable {
		t.Errorf("x+0: want NodeVariable, got %v", got)
	}
}

func TestSimplify_AddZeroRight_Const(t *testing.T) {
	// 3 + 0 → Const(3)
	tree := expression.NewBinary(expression.NodeAdd,
		expression.NewConstant(3), expression.NewConstant(0))
	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeConstant || got.Value != 3 {
		t.Errorf("3+0: want Const(3), got %v", got)
	}
}

func TestSimplify_AddZeroRight_NestedChild(t *testing.T) {
	// (x * 1) + 0: both rules fire bottom-up → result is NodeVariable
	inner := expression.NewBinary(expression.NodeMul,
		expression.NewVariable(), expression.NewConstant(1))
	tree := expression.NewBinary(expression.NodeAdd, inner, expression.NewConstant(0))
	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeVariable {
		t.Errorf("(x*1)+0: want NodeVariable after chained reduction, got %v", got)
	}
}

func TestSimplify_EvalConsistency_Add(t *testing.T) {
	tree := expression.NewBinary(expression.NodeAdd,
		expression.NewVariable(), expression.NewConstant(0))
	checkEvalConsistent(t, "x+0", tree, simplifyXRange)
}

// ─── Regla: x - 0 → x ─────────────────────────────────────────────────────

func TestSimplify_SubZeroRight_Var(t *testing.T) {
	tree := expression.NewBinary(expression.NodeSub,
		expression.NewVariable(), expression.NewConstant(0))
	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeVariable {
		t.Errorf("x-0: want NodeVariable, got %v", got)
	}
}

func TestSimplify_SubZeroRight_Const(t *testing.T) {
	// 5 - 0 → Const(5)
	tree := expression.NewBinary(expression.NodeSub,
		expression.NewConstant(5), expression.NewConstant(0))
	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeConstant || got.Value != 5 {
		t.Errorf("5-0: want Const(5), got %v", got)
	}
}

func TestSimplify_EvalConsistency_Sub(t *testing.T) {
	tree := expression.NewBinary(expression.NodeSub,
		expression.NewVariable(), expression.NewConstant(0))
	checkEvalConsistent(t, "x-0", tree, simplifyXRange)
}

// ─── Regla: x * 1 → x ─────────────────────────────────────────────────────

func TestSimplify_MulOneRight_Var(t *testing.T) {
	tree := expression.NewBinary(expression.NodeMul,
		expression.NewVariable(), expression.NewConstant(1))
	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeVariable {
		t.Errorf("x*1: want NodeVariable, got %v", got)
	}
}

func TestSimplify_MulOneRight_Const(t *testing.T) {
	// 7 * 1 → Const(7)
	tree := expression.NewBinary(expression.NodeMul,
		expression.NewConstant(7), expression.NewConstant(1))
	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeConstant || got.Value != 7 {
		t.Errorf("7*1: want Const(7), got %v", got)
	}
}

func TestSimplify_EvalConsistency_MulOne(t *testing.T) {
	tree := expression.NewBinary(expression.NodeMul,
		expression.NewVariable(), expression.NewConstant(1))
	checkEvalConsistent(t, "x*1", tree, simplifyXRange)
}

// ─── Regla: x / 1 → x ─────────────────────────────────────────────────────

func TestSimplify_DivOneRight_Var(t *testing.T) {
	tree := expression.NewBinary(expression.NodeDiv,
		expression.NewVariable(), expression.NewConstant(1))
	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeVariable {
		t.Errorf("x/1: want NodeVariable, got %v", got)
	}
}

func TestSimplify_DivOneRight_Const(t *testing.T) {
	// 9 / 1 → Const(9)
	tree := expression.NewBinary(expression.NodeDiv,
		expression.NewConstant(9), expression.NewConstant(1))
	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeConstant || got.Value != 9 {
		t.Errorf("9/1: want Const(9), got %v", got)
	}
}

func TestSimplify_EvalConsistency_DivOne(t *testing.T) {
	tree := expression.NewBinary(expression.NodeDiv,
		expression.NewVariable(), expression.NewConstant(1))
	checkEvalConsistent(t, "x/1", tree, simplifyXRange)
}

// ─── Regla: 0 / c → 0 (solo con constante no-nula finita) ────────────────

func TestSimplify_DivZeroLeft_PositiveDen(t *testing.T) {
	// 0 / 3 → Const(0)
	tree := expression.NewBinary(expression.NodeDiv,
		expression.NewConstant(0), expression.NewConstant(3))
	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeConstant || got.Value != 0 {
		t.Errorf("0/3: want Const(0), got %v", got)
	}
}

func TestSimplify_DivZeroLeft_NegativeDen(t *testing.T) {
	// 0 / -5 → Const(0)
	tree := expression.NewBinary(expression.NodeDiv,
		expression.NewConstant(0), expression.NewConstant(-5))
	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeConstant || got.Value != 0 {
		t.Errorf("0/(-5): want Const(0), got %v", got)
	}
}

func TestSimplify_DivZeroLeft_ZeroDen_NoChange(t *testing.T) {
	// 0 / 0 → debe permanecer como NodeDiv (div-by-zero preservado)
	tree := expression.NewBinary(expression.NodeDiv,
		expression.NewConstant(0), expression.NewConstant(0))
	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeDiv {
		t.Errorf("0/0: regla NO debe aplicarse, want NodeDiv, got Type=%v", got.Type)
	}
	// Ambos deben producir EvalError
	_, origErr := expression.Evaluate(tree, 0)
	_, simpErr := expression.Evaluate(got, 0)
	if origErr == nil || simpErr == nil {
		t.Errorf("0/0: se esperaba EvalError en ambos — orig=%v simp=%v", origErr, simpErr)
	}
}

func TestSimplify_DivZeroLeft_VarDen_NoChange(t *testing.T) {
	// 0 / x → NO simplificar (x podría ser 0)
	tree := expression.NewBinary(expression.NodeDiv,
		expression.NewConstant(0), expression.NewVariable())
	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeDiv {
		t.Errorf("0/x: regla NO debe aplicarse, want NodeDiv, got Type=%v", got.Type)
	}
}

func TestSimplify_EvalConsistency_DivZero(t *testing.T) {
	tree := expression.NewBinary(expression.NodeDiv,
		expression.NewConstant(0), expression.NewConstant(5))
	checkEvalConsistent(t, "0/5", tree, simplifyXRange)
}

// ─── Regla: x * 0 → 0 (conservadora: solo hojas seguras) ────────────────

func TestSimplify_MulZero_VarTimesZero(t *testing.T) {
	// x * 0 → Const(0) — Variable es hoja segura
	tree := expression.NewBinary(expression.NodeMul,
		expression.NewVariable(), expression.NewConstant(0))
	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeConstant || got.Value != 0 {
		t.Errorf("x*0: want Const(0), got %v", got)
	}
}

func TestSimplify_MulZero_ZeroTimesVar(t *testing.T) {
	// 0 * x → Const(0)
	tree := expression.NewBinary(expression.NodeMul,
		expression.NewConstant(0), expression.NewVariable())
	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeConstant || got.Value != 0 {
		t.Errorf("0*x: want Const(0), got %v", got)
	}
}

func TestSimplify_MulZero_FiniteConstTimesZero(t *testing.T) {
	// 3 * 0 → Const(0) — constante finita es hoja segura
	tree := expression.NewBinary(expression.NodeMul,
		expression.NewConstant(3), expression.NewConstant(0))
	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeConstant || got.Value != 0 {
		t.Errorf("3*0: want Const(0), got %v", got)
	}
}

func TestSimplify_MulZero_ZeroTimesFiniteConst(t *testing.T) {
	// 0 * 3 → Const(0)
	tree := expression.NewBinary(expression.NodeMul,
		expression.NewConstant(0), expression.NewConstant(3))
	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeConstant || got.Value != 0 {
		t.Errorf("0*3: want Const(0), got %v", got)
	}
}

func TestSimplify_MulZero_SubtreeLeft_NoChange(t *testing.T) {
	// (1/0) * 0 — el subárbol izquierdo no es hoja segura;
	// la regla NO debe aplicarse para preservar EvalError.
	inner := expression.NewBinary(expression.NodeDiv,
		expression.NewConstant(1), expression.NewConstant(0))
	tree := expression.NewBinary(expression.NodeMul, inner, expression.NewConstant(0))
	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeMul {
		t.Errorf("(1/0)*0: regla NO debe aplicarse, want NodeMul, got Type=%v", got.Type)
	}
	_, origErr := expression.Evaluate(tree, 1)
	_, simpErr := expression.Evaluate(got, 1)
	if (origErr != nil) != (simpErr != nil) {
		t.Errorf("(1/0)*0: comportamiento de error cambió — orig=%v simp=%v", origErr, simpErr)
	}
}

func TestSimplify_MulZero_SubtreeRight_NoChange(t *testing.T) {
	// 0 * (1/0) — el subárbol derecho no es hoja segura
	inner := expression.NewBinary(expression.NodeDiv,
		expression.NewConstant(1), expression.NewConstant(0))
	tree := expression.NewBinary(expression.NodeMul, expression.NewConstant(0), inner)
	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeMul {
		t.Errorf("0*(1/0): regla NO debe aplicarse, want NodeMul, got Type=%v", got.Type)
	}
	_, origErr := expression.Evaluate(tree, 1)
	_, simpErr := expression.Evaluate(got, 1)
	if (origErr != nil) != (simpErr != nil) {
		t.Errorf("0*(1/0): comportamiento de error cambió — orig=%v simp=%v", origErr, simpErr)
	}
}

func TestSimplify_MulZero_AddSubtree_NoChange(t *testing.T) {
	// (x + 3) * 0 — NodeAdd no es una hoja segura por la regla conservadora.
	// Documenta el límite intencional de la implementación.
	inner := expression.NewBinary(expression.NodeAdd,
		expression.NewVariable(), expression.NewConstant(3))
	tree := expression.NewBinary(expression.NodeMul, inner, expression.NewConstant(0))
	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeMul {
		t.Errorf("(x+3)*0: regla conservadora: want NodeMul, got Type=%v", got.Type)
	}
}

func TestSimplify_MulZero_NaNConst_NoChange(t *testing.T) {
	// NaN * 0: NaN no es una constante finita → no es hoja segura
	tree := expression.NewBinary(expression.NodeMul,
		expression.NewConstant(math.NaN()), expression.NewConstant(0))
	got := expression.Simplify(tree)
	if got.Type != expression.NodeMul {
		t.Errorf("NaN*0: constante no finita → regla NO aplica, want NodeMul, got %v", got.Type)
	}
}

func TestSimplify_MulZero_InfConst_NoChange(t *testing.T) {
	// +Inf * 0: Inf no es una constante finita → no es hoja segura
	tree := expression.NewBinary(expression.NodeMul,
		expression.NewConstant(math.Inf(1)), expression.NewConstant(0))
	got := expression.Simplify(tree)
	if got.Type != expression.NodeMul {
		t.Errorf("+Inf*0: constante no finita → regla NO aplica, want NodeMul, got %v", got.Type)
	}
}

func TestSimplify_EvalConsistency_MulZeroVar(t *testing.T) {
	tree := expression.NewBinary(expression.NodeMul,
		expression.NewVariable(), expression.NewConstant(0))
	checkEvalConsistent(t, "x*0", tree, simplifyXRange)
}

func TestSimplify_EvalConsistency_MulZeroConst(t *testing.T) {
	tree := expression.NewBinary(expression.NodeMul,
		expression.NewConstant(3), expression.NewConstant(0))
	checkEvalConsistent(t, "3*0", tree, simplifyXRange)
}

// ─── Casos sin regla aplicable ─────────────────────────────────────────────

func TestSimplify_NoRule_AddNonZero(t *testing.T) {
	tree := expression.NewBinary(expression.NodeAdd,
		expression.NewVariable(), expression.NewConstant(3))
	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeAdd {
		t.Errorf("x+3: sin regla, want NodeAdd, got Type=%v", got.Type)
	}
	if got == tree {
		t.Error("x+3: debe retornar un nodo nuevo, no el mismo puntero")
	}
}

func TestSimplify_NoRule_MulNonOne(t *testing.T) {
	tree := expression.NewBinary(expression.NodeMul,
		expression.NewVariable(), expression.NewConstant(3))
	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeMul {
		t.Errorf("x*3: sin regla, want NodeMul, got Type=%v", got.Type)
	}
}

func TestSimplify_NoRule_Pow(t *testing.T) {
	// NodePow no tiene reglas de simplificación en Fase 13
	cases := []*expression.Node{
		expression.NewBinary(expression.NodePow, expression.NewVariable(), expression.NewConstant(0)),
		expression.NewBinary(expression.NodePow, expression.NewVariable(), expression.NewConstant(1)),
		expression.NewBinary(expression.NodePow, expression.NewVariable(), expression.NewConstant(2)),
	}
	for _, c := range cases {
		got := expression.Simplify(c)
		if got == nil || got.Type != expression.NodePow {
			t.Errorf("Pow: sin regla, want NodePow, got %v", got)
		}
	}
}

// ─── Nil y hojas ──────────────────────────────────────────────────────────

func TestSimplify_Nil(t *testing.T) {
	if expression.Simplify(nil) != nil {
		t.Error("Simplify(nil) debe retornar nil")
	}
}

func TestSimplify_Leaf_Variable(t *testing.T) {
	tree := expression.NewVariable()
	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeVariable {
		t.Errorf("Simplify(x): want NodeVariable, got %v", got)
	}
	if got == tree {
		t.Error("Simplify(x): debe retornar un nodo nuevo")
	}
}

func TestSimplify_Leaf_Const(t *testing.T) {
	tree := expression.NewConstant(5)
	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeConstant || got.Value != 5 {
		t.Errorf("Simplify(5): want Const(5), got %v", got)
	}
	if got == tree {
		t.Error("Simplify(5): debe retornar un nodo nuevo")
	}
}

// ─── Nodos unarios ────────────────────────────────────────────────────────

func TestSimplify_Unary_NoRules(t *testing.T) {
	// Los nodos unarios no tienen reglas; retornan copia fresca con mismo tipo
	unaries := []*expression.Node{
		expression.NewUnary(expression.NodeSqrt, expression.NewVariable()),
		expression.NewUnary(expression.NodeLn, expression.NewVariable()),
		expression.NewUnary(expression.NodeAbs, expression.NewVariable()),
		expression.NewUnary(expression.NodeExp, expression.NewVariable()),
		expression.NewUnary(expression.NodeSin, expression.NewVariable()),
		expression.NewUnary(expression.NodeCos, expression.NewVariable()),
	}
	for _, u := range unaries {
		got := expression.Simplify(u)
		if got == nil || got.Type != u.Type {
			t.Errorf("unary %v: want mismo tipo, got %v", u.Type, got)
		}
	}
}

func TestSimplify_Unary_ChildSimplified(t *testing.T) {
	// ln(x + 0) → ln(x): la regla se aplica al hijo aunque ln no tenga regla
	tree := expression.NewUnary(expression.NodeLn,
		expression.NewBinary(expression.NodeAdd,
			expression.NewVariable(), expression.NewConstant(0)))
	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeLn {
		t.Fatalf("ln(x+0): want NodeLn, got %v", got)
	}
	if got.Left == nil || got.Left.Type != expression.NodeVariable {
		t.Errorf("ln(x+0): hijo debe ser NodeVariable, got %v", got.Left)
	}
}

func TestSimplify_Unary_SqrtChildSimplified(t *testing.T) {
	// sqrt(x * 1) → sqrt(x)
	tree := expression.NewUnary(expression.NodeSqrt,
		expression.NewBinary(expression.NodeMul,
			expression.NewVariable(), expression.NewConstant(1)))
	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeSqrt {
		t.Fatalf("sqrt(x*1): want NodeSqrt, got %v", got)
	}
	if got.Left == nil || got.Left.Type != expression.NodeVariable {
		t.Errorf("sqrt(x*1): hijo debe ser NodeVariable, got %v", got.Left)
	}
}

// ─── Anidamiento y composición de reglas ──────────────────────────────────

func TestSimplify_Nested_MultipleRules(t *testing.T) {
	// Add(Mul(x, 1), Sub(Add(x, 0), 0))
	//   bottom-up: Mul(x,1)→x; Add(x,0)→x; Sub(x,0)→x
	//   result: Add(x, x)
	inner1 := expression.NewBinary(expression.NodeMul,
		expression.NewVariable(), expression.NewConstant(1))
	inner2 := expression.NewBinary(expression.NodeSub,
		expression.NewBinary(expression.NodeAdd,
			expression.NewVariable(), expression.NewConstant(0)),
		expression.NewConstant(0))
	tree := expression.NewBinary(expression.NodeAdd, inner1, inner2)

	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeAdd {
		t.Fatalf("anidado: want NodeAdd, got %v", got)
	}
	if got.Left == nil || got.Left.Type != expression.NodeVariable {
		t.Errorf("anidado: izquierdo debe ser NodeVariable, got %v", got.Left)
	}
	if got.Right == nil || got.Right.Type != expression.NodeVariable {
		t.Errorf("anidado: derecho debe ser NodeVariable, got %v", got.Right)
	}
}

func TestSimplify_Chained_DeepReduction(t *testing.T) {
	// Mul(Add(x, 0), Const(1))
	//   bottom-up: Add(x,0)→x; Mul(x,1)→x
	//   result: NodeVariable
	inner := expression.NewBinary(expression.NodeAdd,
		expression.NewVariable(), expression.NewConstant(0))
	tree := expression.NewBinary(expression.NodeMul, inner, expression.NewConstant(1))
	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeVariable {
		t.Errorf("(x+0)*1: want NodeVariable después de reducción doble, got %v", got)
	}
}

func TestSimplify_EvalConsistency_Nested(t *testing.T) {
	// (x + 0) * 1 → x
	tree := expression.NewBinary(expression.NodeMul,
		expression.NewBinary(expression.NodeAdd,
			expression.NewVariable(), expression.NewConstant(0)),
		expression.NewConstant(1))
	checkEvalConsistent(t, "(x+0)*1", tree, simplifyXRange)
}

// ─── Independencia del original ────────────────────────────────────────────

func TestSimplify_OriginalUnchanged(t *testing.T) {
	tree := expression.NewBinary(expression.NodeAdd,
		expression.NewVariable(), expression.NewConstant(0))
	fmtBefore := expression.Format(tree)
	_ = expression.Simplify(tree)
	fmtAfter := expression.Format(tree)
	if fmtBefore != fmtAfter {
		t.Errorf("Simplify modificó el árbol original: antes=%q, después=%q", fmtBefore, fmtAfter)
	}
}

func TestSimplify_ResultIsIndependent(t *testing.T) {
	// Mutar el resultado no debe afectar el original
	tree := expression.NewBinary(expression.NodeMul,
		expression.NewVariable(), expression.NewConstant(1))
	got := expression.Simplify(tree) // → NodeVariable
	if got == nil {
		t.Fatal("Simplify devolvió nil inesperadamente")
	}
	got.Type = expression.NodeConstant
	got.Value = 999
	if tree.Type != expression.NodeMul {
		t.Error("mutar el resultado de Simplify afectó el árbol original")
	}
}

func TestSimplify_NoRuleResultIsIndependent(t *testing.T) {
	// Sin regla: la copia devuelta debe ser independiente del original
	tree := expression.NewBinary(expression.NodeAdd,
		expression.NewVariable(), expression.NewConstant(5))
	got := expression.Simplify(tree)
	if got == nil || got.Type != expression.NodeAdd {
		t.Fatal("resultado inesperado para x+5")
	}
	if got.Left != nil {
		got.Left.Type = expression.NodeConstant
		got.Left.Value = 42
	}
	if tree.Left == nil || tree.Left.Type != expression.NodeVariable {
		t.Error("mutar el hijo del resultado afectó el árbol original")
	}
}

// ─── Idempotencia ─────────────────────────────────────────────────────────

func TestSimplify_Idempotent(t *testing.T) {
	cases := []*expression.Node{
		expression.NewBinary(expression.NodeAdd, expression.NewVariable(), expression.NewConstant(0)),
		expression.NewBinary(expression.NodeSub, expression.NewVariable(), expression.NewConstant(0)),
		expression.NewBinary(expression.NodeMul, expression.NewVariable(), expression.NewConstant(1)),
		expression.NewBinary(expression.NodeDiv, expression.NewVariable(), expression.NewConstant(1)),
		expression.NewBinary(expression.NodeMul, expression.NewVariable(), expression.NewConstant(0)),
		expression.NewBinary(expression.NodeDiv, expression.NewConstant(0), expression.NewConstant(3)),
		// Compuesto: doble reducción
		expression.NewBinary(expression.NodeAdd,
			expression.NewBinary(expression.NodeMul,
				expression.NewVariable(), expression.NewConstant(1)),
			expression.NewConstant(0)),
		// Sin regla
		expression.NewBinary(expression.NodeAdd, expression.NewVariable(), expression.NewConstant(3)),
		expression.NewVariable(),
		expression.NewConstant(7),
	}
	for i, tree := range cases {
		once := expression.Simplify(tree)
		twice := expression.Simplify(once)
		s1 := expression.Format(once)
		s2 := expression.Format(twice)
		if s1 != s2 {
			t.Errorf("caso %d: no idempotente: Simplify=%q, Simplify(Simplify)=%q", i, s1, s2)
		}
	}
}

// ─── Preservación de EvalError ────────────────────────────────────────────

func TestSimplify_EvalError_DivByZeroVar(t *testing.T) {
	// 0 / x: no se simplifica; en x=0 produce EvalError
	tree := expression.NewBinary(expression.NodeDiv,
		expression.NewConstant(0), expression.NewVariable())
	got := expression.Simplify(tree)
	if got.Type != expression.NodeDiv {
		t.Errorf("0/x: NO debe simplificarse, want NodeDiv, got %v", got)
	}
	_, origErr := expression.Evaluate(tree, 0)
	_, simpErr := expression.Evaluate(got, 0)
	if origErr == nil || simpErr == nil {
		t.Errorf("0/x en x=0: ambos deben dar EvalError — orig=%v simp=%v", origErr, simpErr)
	}
}

func TestSimplify_EvalError_MulZeroSubtree(t *testing.T) {
	// (1/0) * 0: el EvalError del subárbol izquierdo debe preservarse
	inner := expression.NewBinary(expression.NodeDiv,
		expression.NewConstant(1), expression.NewConstant(0))
	tree := expression.NewBinary(expression.NodeMul, inner, expression.NewConstant(0))
	got := expression.Simplify(tree)
	_, origErr := expression.Evaluate(tree, 0)
	_, simpErr := expression.Evaluate(got, 0)
	if origErr == nil {
		t.Fatal("original (1/0)*0 debe producir EvalError")
	}
	if simpErr == nil {
		t.Error("simplificado (1/0)*0 debe seguir produciendo EvalError (regla no debe aplicarse)")
	}
}
