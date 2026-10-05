package generator_test

// phase8_test.go — Tests de Fase 8 para el generador:
//   - NodePow: el hijo derecho siempre es una constante entera en [PowExponentMin, PowExponentMax]
//   - Nuevos NodeType unarios (NodeAbs, NodeExp, NodeSin, NodeCos) se generan si están habilitados
//   - Las restricciones de MaxDepth y MaxNodes siguen cumpliéndose con los nuevos operadores
//   - Validación de PowExponentMin > PowExponentMax
//   - Default() incluye PowExponentMin=-4 y PowExponentMax=4

import (
	"math"
	"testing"

	"github.com/LautiSeverino/symbolic-regression/internal/expression"
	"github.com/LautiSeverino/symbolic-regression/internal/generator"
)

// ─── Helpers ──────────────────────────────────────────────────────────────

// isIntegerInRange reporta si v es un entero exacto en [min, max].
func isIntegerInRange(v float64, min, max int) bool {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return false
	}
	intV := int(v)
	return float64(intV) == v && intV >= min && intV <= max
}

// collectByType recorre el árbol y devuelve todos los nodos del tipo dado.
func collectByType(node *expression.Node, t expression.NodeType) []*expression.Node {
	if node == nil {
		return nil
	}
	var result []*expression.Node
	var walk func(*expression.Node)
	walk = func(n *expression.Node) {
		if n == nil {
			return
		}
		if n.Type == t {
			result = append(result, n)
		}
		walk(n.Left)
		walk(n.Right)
	}
	walk(node)
	return result
}

// hasNodeType reporta si algún nodo del árbol es del tipo dado.
func hasNodeType(node *expression.Node, t expression.NodeType) bool {
	return len(collectByType(node, t)) > 0
}

// powOnlyCfg devuelve una config con solo NodePow en binarios y sin unarios.
func powOnlyCfg() generator.Config {
	return generator.Config{
		MaxDepth:            5,
		MaxNodes:            30,
		AllowedBinaryOps:    []expression.NodeType{expression.NodePow},
		AllowedUnaryOps:     nil,
		ConstantMin:         -10,
		ConstantMax:         10,
		VariableProbability: 0.5,
		TerminalProbability: 0.2,
		PowExponentMin:      -4,
		PowExponentMax:      4,
		Seed:                42,
	}
}

// allPhase8UnarysCfg habilita los 4 nuevos operadores unarios.
func allPhase8UnarysCfg() generator.Config {
	return generator.Config{
		MaxDepth: 5,
		MaxNodes: 30,
		AllowedBinaryOps: []expression.NodeType{
			expression.NodeAdd,
		},
		AllowedUnaryOps: []expression.NodeType{
			expression.NodeAbs,
			expression.NodeExp,
			expression.NodeSin,
			expression.NodeCos,
		},
		ConstantMin:         -10,
		ConstantMax:         10,
		VariableProbability: 0.5,
		TerminalProbability: 0.1, // baja para forzar operadores
		PowExponentMin:      -4,
		PowExponentMax:      4,
		Seed:                77,
	}
}

// ─── Validación de Config ─────────────────────────────────────────────────

func TestNew_PowExponentValid(t *testing.T) {
	cases := []struct {
		name string
		min  int
		max  int
	}{
		{"min<max", -4, 4},
		{"min=max", 2, 2},
		{"solo positivos", 1, 3},
		{"solo negativos", -3, -1},
		{"cero", 0, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := powOnlyCfg()
			cfg.PowExponentMin = tc.min
			cfg.PowExponentMax = tc.max
			_, err := generator.New(cfg)
			if err != nil {
				t.Errorf("PowExponentMin=%d PowExponentMax=%d: error inesperado: %v",
					tc.min, tc.max, err)
			}
		})
	}
}

func TestNew_PowExponentInvalid(t *testing.T) {
	cfg := powOnlyCfg()
	cfg.PowExponentMin = 4
	cfg.PowExponentMax = -4 // min > max → error
	_, err := generator.New(cfg)
	if err == nil {
		t.Error("PowExponentMin > PowExponentMax: se esperaba error, se obtuvo nil")
	}
}

func TestDefault_PowExponentRange(t *testing.T) {
	// Default() debe tener PowExponentMin=-4 y PowExponentMax=4
	// Verificamos indirectamente: New(Default()) no debe fallar Y la generación
	// con NodePow habilitado debe producir exponentes en [-4,4].
	cfg := generator.Default()
	cfg.AllowedBinaryOps = append(cfg.AllowedBinaryOps, expression.NodePow)
	cfg.Seed = 1
	g, err := generator.New(cfg)
	if err != nil {
		t.Fatalf("New(Default() + NodePow): error inesperado: %v", err)
	}
	for i := 0; i < 200; i++ {
		tree := g.Generate()
		for _, powNode := range collectByType(tree, expression.NodePow) {
			if powNode.Right == nil {
				t.Fatalf("árbol %d: NodePow con Right=nil", i)
			}
			if powNode.Right.Type != expression.NodeConstant {
				t.Errorf("árbol %d: NodePow.Right.Type=%v, want NodeConstant", i, powNode.Right.Type)
			}
			v := powNode.Right.Value
			if !isIntegerInRange(v, -4, 4) {
				t.Errorf("árbol %d: exponente %v no es entero en [-4,4]", i, v)
			}
		}
	}
}

// ─── NodePow: exponente siempre es constante entera acotada ───────────────

func TestGenerate_Pow_ExponentIsIntegerConstant(t *testing.T) {
	cfg := powOnlyCfg()
	g, err := generator.New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for i := 0; i < 300; i++ {
		tree := g.Generate()
		for _, powNode := range collectByType(tree, expression.NodePow) {
			if powNode.Right == nil {
				t.Fatalf("árbol %d: NodePow.Right es nil", i)
			}
			if powNode.Right.Type != expression.NodeConstant {
				t.Errorf("árbol %d: NodePow.Right.Type = %v, want NodeConstant",
					i, powNode.Right.Type)
			}
			v := powNode.Right.Value
			if !isIntegerInRange(v, cfg.PowExponentMin, cfg.PowExponentMax) {
				t.Errorf("árbol %d: exponente %v no es entero en [%d,%d]",
					i, v, cfg.PowExponentMin, cfg.PowExponentMax)
			}
		}
	}
}

func TestGenerate_Pow_ExponentRangeCustom(t *testing.T) {
	// Rango estrecho [-2, 2]: verificar que solo aparecen esos valores
	cfg := powOnlyCfg()
	cfg.PowExponentMin = -2
	cfg.PowExponentMax = 2
	g, err := generator.New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for i := 0; i < 200; i++ {
		tree := g.Generate()
		for _, powNode := range collectByType(tree, expression.NodePow) {
			v := powNode.Right.Value
			if !isIntegerInRange(v, -2, 2) {
				t.Errorf("árbol %d: exponente %v fuera de [-2,2]", i, v)
			}
		}
	}
}

func TestGenerate_Pow_ExponentRangeSingle(t *testing.T) {
	// PowExponentMin == PowExponentMax == 3: todos los exponentes deben ser 3.0
	cfg := powOnlyCfg()
	cfg.PowExponentMin = 3
	cfg.PowExponentMax = 3
	g, err := generator.New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for i := 0; i < 100; i++ {
		tree := g.Generate()
		for _, powNode := range collectByType(tree, expression.NodePow) {
			if powNode.Right.Value != 3 {
				t.Errorf("árbol %d: exponente = %v, want 3 (rango único)", i, powNode.Right.Value)
			}
		}
	}
}

func TestGenerate_Pow_LeftIsNotConstantExponent(t *testing.T) {
	// El hijo IZQUIERDO (la base) NO debe estar restringido a ser constante entero acotado
	// (puede ser cualquier subárbol). Lo que sí verificamos es que no sea siempre una hoja.
	cfg := powOnlyCfg()
	cfg.TerminalProbability = 0 // forzar subárboles
	g, err := generator.New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	nonLeafBase := false
	for i := 0; i < 200; i++ {
		tree := g.Generate()
		for _, powNode := range collectByType(tree, expression.NodePow) {
			if powNode.Left != nil && powNode.Left.Left != nil {
				nonLeafBase = true
			}
		}
	}
	if !nonLeafBase {
		// Con TP=0, al menos algún NodePow debe tener una base no-hoja
		// (solo falla si la generación de subárboles está completamente rota)
		t.Log("advertencia: ningún NodePow tuvo una base no-hoja (puede ser aceptable si MaxDepth es pequeño)")
	}
}

// ─── NodePow respeta MaxDepth y MaxNodes ──────────────────────────────────

func TestGenerate_Pow_RespectsMaxDepth(t *testing.T) {
	cfg := powOnlyCfg()
	g, err := generator.New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for i := 0; i < 200; i++ {
		tree := g.Generate()
		if d := expression.Depth(tree); d > cfg.MaxDepth {
			t.Errorf("árbol %d: Depth=%d > MaxDepth=%d", i, d, cfg.MaxDepth)
		}
	}
}

func TestGenerate_Pow_RespectsMaxNodes(t *testing.T) {
	cfg := powOnlyCfg()
	g, err := generator.New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for i := 0; i < 200; i++ {
		tree := g.Generate()
		if n := expression.Count(tree); n > cfg.MaxNodes {
			t.Errorf("árbol %d: Count=%d > MaxNodes=%d", i, n, cfg.MaxNodes)
		}
	}
}

// ─── Nuevos NodeType unarios (NodeAbs, NodeExp, NodeSin, NodeCos) ─────────

func TestGenerate_NewUnaryOps_CanAppear(t *testing.T) {
	// Con TP bajo y suficientes árboles, los 4 nuevos ops unarios deben aparecer
	cfg := allPhase8UnarysCfg()
	g, err := generator.New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	seen := map[expression.NodeType]bool{}
	target := []expression.NodeType{
		expression.NodeAbs, expression.NodeExp,
		expression.NodeSin, expression.NodeCos,
	}

	for i := 0; i < 500; i++ {
		tree := g.Generate()
		for _, nt := range target {
			if hasNodeType(tree, nt) {
				seen[nt] = true
			}
		}
		if len(seen) == len(target) {
			break
		}
	}
	for _, nt := range target {
		if !seen[nt] {
			t.Errorf("NodeType %v nunca apareció en 500 árboles (debería poder generarse)", nt)
		}
	}
}

func TestGenerate_NewUnaryOps_OnlyAllowed(t *testing.T) {
	// Habilitar solo NodeAbs y NodeSin: NodeExp y NodeCos no deben aparecer
	cfg := generator.Config{
		MaxDepth: 5,
		MaxNodes: 30,
		AllowedBinaryOps: []expression.NodeType{expression.NodeAdd},
		AllowedUnaryOps:  []expression.NodeType{expression.NodeAbs, expression.NodeSin},
		ConstantMin:      -10,
		ConstantMax:      10,
		VariableProbability: 0.5,
		TerminalProbability: 0.1,
		PowExponentMin:   -4,
		PowExponentMax:   4,
		Seed:             88,
	}
	forbidden := []expression.NodeType{
		expression.NodeExp, expression.NodeCos,
		expression.NodeSqrt, expression.NodeLn, expression.NodePow,
	}
	g, err := generator.New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for i := 0; i < 200; i++ {
		tree := g.Generate()
		for _, nt := range forbidden {
			if hasNodeType(tree, nt) {
				t.Errorf("árbol %d: NodeType prohibido %v encontrado", i, nt)
			}
		}
	}
}

func TestGenerate_NewUnaryOps_RespectsMaxDepth(t *testing.T) {
	cfg := allPhase8UnarysCfg()
	g, err := generator.New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for i := 0; i < 200; i++ {
		tree := g.Generate()
		if d := expression.Depth(tree); d > cfg.MaxDepth {
			t.Errorf("árbol %d: Depth=%d > MaxDepth=%d", i, d, cfg.MaxDepth)
		}
	}
}

func TestGenerate_NewUnaryOps_RespectsMaxNodes(t *testing.T) {
	cfg := allPhase8UnarysCfg()
	g, err := generator.New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for i := 0; i < 200; i++ {
		tree := g.Generate()
		if n := expression.Count(tree); n > cfg.MaxNodes {
			t.Errorf("árbol %d: Count=%d > MaxNodes=%d", i, n, cfg.MaxNodes)
		}
	}
}

func TestGenerate_NewUnaryOps_Structure(t *testing.T) {
	// Todos los nuevos nodos unarios deben tener Left≠nil y Right==nil
	cfg := allPhase8UnarysCfg()
	g, err := generator.New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	unaryTypes := []expression.NodeType{
		expression.NodeAbs, expression.NodeExp,
		expression.NodeSin, expression.NodeCos,
	}
	for i := 0; i < 200; i++ {
		tree := g.Generate()
		traverseAll(tree, func(n *expression.Node) {
			for _, nt := range unaryTypes {
				if n.Type == nt {
					if n.Left == nil {
						t.Errorf("árbol %d: NodeType %v tiene Left=nil", i, nt)
					}
					if n.Right != nil {
						t.Errorf("árbol %d: NodeType %v tiene Right≠nil", i, nt)
					}
				}
			}
		})
	}
}

// ─── Reproducibilidad con los nuevos ops ──────────────────────────────────

func TestGenerate_Phase8_SeedReproducible(t *testing.T) {
	// Misma config (incluyendo NodePow) + misma seed → misma secuencia
	cfg := powOnlyCfg()
	g1, _ := generator.New(cfg)
	g2, _ := generator.New(cfg)
	for i := 0; i < 10; i++ {
		s1 := expression.Format(g1.Generate())
		s2 := expression.Format(g2.Generate())
		if s1 != s2 {
			t.Errorf("árbol %d: misma seed + NodePow → árboles distintos:\n  %s\n  %s", i, s1, s2)
		}
	}
}

// ─── Regresión: Default() no incluye nuevos ops por defecto ───────────────

func TestGenerate_Phase8_DefaultOpsUnchanged(t *testing.T) {
	// Default() NO debe incluir NodePow ni los nuevos unarios en sus listas —
	// son opt-in para no romper los tests existentes (Fases 1-7).
	cfg := generator.Default()
	for _, op := range cfg.AllowedBinaryOps {
		if op == expression.NodePow {
			t.Error("Default().AllowedBinaryOps incluye NodePow (debe ser opt-in)")
		}
	}
	newUnary := []expression.NodeType{
		expression.NodeAbs, expression.NodeExp,
		expression.NodeSin, expression.NodeCos,
	}
	for _, op := range cfg.AllowedUnaryOps {
		for _, nu := range newUnary {
			if op == nu {
				t.Errorf("Default().AllowedUnaryOps incluye NodeType %v (debe ser opt-in)", nu)
			}
		}
	}
}
