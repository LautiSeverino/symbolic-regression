package generator_test

// robustness_test.go — Fase 9: robustez del pipeline generator → evaluator.
//
// Verifica que cualquier árbol producido por el generador pueda evaluarse
// sin pánico, con un rango amplio de x y con todas las configuraciones
// de operadores disponibles.
//
// No reemplaza los tests de expresión individual (internal/expression/robustness_test.go);
// cubre el camino completo: generación aleatoria → evaluación numérica.

import (
	"math"
	"testing"

	"github.com/LautiSeverino/symbolic-regression/internal/expression"
	"github.com/LautiSeverino/symbolic-regression/internal/generator"
)

// xRange es el conjunto de valores de x sobre los que se evalúan los árboles.
// Incluye cero, negativos, positivos, y valores extremos para ejercitar
// todos los dominios numéricos posibles.
var xRange = []float64{
	-1e6, -100, -10, -3, -1, -0.5, -0.1,
	0,
	0.1, 0.5, 1, 3, 10, 100, 1e6,
}

// fullPhase8Config devuelve una configuración con todos los operadores
// implementados hasta Fase 8, incluyendo NodePow y los cuatro nuevos unarios.
func fullPhase8Config(seed int64) generator.Config {
	return generator.Config{
		MaxDepth: 5,
		MaxNodes: 30,
		AllowedBinaryOps: []expression.NodeType{
			expression.NodeAdd, expression.NodeSub,
			expression.NodeMul, expression.NodeDiv,
			expression.NodePow,
		},
		AllowedUnaryOps: []expression.NodeType{
			expression.NodeSqrt, expression.NodeLn,
			expression.NodeAbs, expression.NodeExp,
			expression.NodeSin, expression.NodeCos,
		},
		ConstantMin:         -10,
		ConstantMax:         10,
		VariableProbability: 0.5,
		TerminalProbability: 0.2,
		PowExponentMin:      -4,
		PowExponentMax:      4,
		Seed:                seed,
	}
}

// ══════════════════════════════════════════════════════════════════════════
// 1. No-panic: árboles generados × rango de x
// ══════════════════════════════════════════════════════════════════════════

func TestGenEval_NoPanic_FullPhase8(t *testing.T) {
	// 500 árboles con el set completo de operadores, evaluados en xRange.
	// Cualquier EvalError es aceptable; los pánico no lo son.
	g, err := generator.New(fullPhase8Config(42))
	if err != nil {
		t.Fatalf("generator.New: %v", err)
	}
	for i := 0; i < 500; i++ {
		tree := g.Generate()
		if tree == nil {
			t.Errorf("árbol %d: Generate() devolvió nil", i)
			continue
		}
		for _, x := range xRange {
			i, x := i, x
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Errorf("árbol %d, x=%v: panic: %v — árbol: %s",
							i, x, r, expression.Format(tree))
					}
				}()
				_, _ = expression.Evaluate(tree, x)
			}()
		}
	}
}

func TestGenEval_NoPanic_BalancedProfile(t *testing.T) {
	// Simula el perfil BALANCED: depth=7, nodes=75.
	// Árboles más grandes, solo operadores básicos + sqrt + ln.
	cfg := generator.Default() // depth=7, nodes=75, AllowedBinary={+,-,*,/}, AllowedUnary={sqrt,ln}
	cfg.Seed = 99
	g, err := generator.New(cfg)
	if err != nil {
		t.Fatalf("generator.New: %v", err)
	}
	for i := 0; i < 300; i++ {
		tree := g.Generate()
		for _, x := range xRange {
			i, x := i, x
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Errorf("árbol %d, x=%v: panic: %v — árbol: %s",
							i, x, r, expression.Format(tree))
					}
				}()
				_, _ = expression.Evaluate(tree, x)
			}()
		}
	}
}

// ══════════════════════════════════════════════════════════════════════════
// 2. Resultados siempre finitos o EvalError — nunca NaN silencioso
// ══════════════════════════════════════════════════════════════════════════

func TestGenEval_ResultFiniteOrError(t *testing.T) {
	// Todo árbol generado debe retornar:
	//   a) un float64 finito, o
	//   b) un EvalError.
	// Nunca debe retornar NaN o ±Inf sin error.
	g, err := generator.New(fullPhase8Config(7))
	if err != nil {
		t.Fatalf("generator.New: %v", err)
	}
	for i := 0; i < 300; i++ {
		tree := g.Generate()
		for _, x := range xRange {
			result, evalErr := expression.Evaluate(tree, x)
			if evalErr != nil {
				// EvalError es aceptable — la capa de fitness lo maneja
				continue
			}
			// Sin error, el resultado debe ser finito
			if math.IsNaN(result) {
				t.Errorf("árbol %d, x=%v: resultado NaN sin EvalError — árbol: %s",
					i, x, expression.Format(tree))
			}
			if math.IsInf(result, 0) {
				t.Errorf("árbol %d, x=%v: resultado Inf sin EvalError — árbol: %s",
					i, x, expression.Format(tree))
			}
		}
	}
}

// ══════════════════════════════════════════════════════════════════════════
// 3. Constantes generadas siempre finitas
// ══════════════════════════════════════════════════════════════════════════

func TestGenEval_GeneratedConstantsFinite(t *testing.T) {
	// El generador nunca debe producir NodeConstant con NaN o ±Inf.
	// Consecuencia: evaluate.go no verá esas constantes en condiciones normales.
	g, err := generator.New(fullPhase8Config(13))
	if err != nil {
		t.Fatalf("generator.New: %v", err)
	}
	for i := 0; i < 200; i++ {
		tree := g.Generate()
		traverseAll(tree, func(n *expression.Node) {
			if n.Type != expression.NodeConstant {
				return
			}
			if math.IsNaN(n.Value) {
				t.Errorf("árbol %d: NodeConstant con valor NaN", i)
			}
			if math.IsInf(n.Value, 0) {
				t.Errorf("árbol %d: NodeConstant con valor Inf (%v)", i, n.Value)
			}
		})
	}
}
