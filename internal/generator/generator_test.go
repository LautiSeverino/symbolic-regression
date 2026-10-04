package generator_test

import (
	"testing"

	"github.com/LautiSeverino/symbolic-regression/internal/expression"
	"github.com/LautiSeverino/symbolic-regression/internal/generator"
)

// ─── Helpers ──────────────────────────────────────────────────────────────

// traverseAll visita todos los nodos del árbol en preorden.
func traverseAll(node *expression.Node, fn func(*expression.Node)) {
	if node == nil {
		return
	}
	fn(node)
	traverseAll(node.Left, fn)
	traverseAll(node.Right, fn)
}

// checkLeavesAtMaxDepth verifica que todos los nodos en expression-depth == maxDepth
// son hojas (sin hijos). Se llama con currentDepth=1 para el nodo raíz.
func checkLeavesAtMaxDepth(t *testing.T, node *expression.Node, maxDepth, currentDepth int) {
	t.Helper()
	if node == nil {
		return
	}
	if currentDepth == maxDepth {
		if node.Left != nil || node.Right != nil {
			t.Errorf("nodo en expression-depth %d no es hoja (Type=%v)", currentDepth, node.Type)
		}
		return // no seguir más profundo (sería > maxDepth)
	}
	checkLeavesAtMaxDepth(t, node.Left, maxDepth, currentDepth+1)
	checkLeavesAtMaxDepth(t, node.Right, maxDepth, currentDepth+1)
}

// ─── Config: New ──────────────────────────────────────────────────────────

func TestNew_Valid(t *testing.T) {
	cases := []struct {
		name string
		cfg  generator.Config
	}{
		{"default", generator.Default()},
		{"MaxDepth=1", func() generator.Config {
			c := generator.Default(); c.MaxDepth = 1; return c
		}()},
		{"sin unary ops", func() generator.Config {
			c := generator.Default(); c.AllowedUnaryOps = nil; return c
		}()},
		{"ambos ops vacios", generator.Config{
			MaxDepth: 3, MaxNodes: 10,
			AllowedBinaryOps: nil, AllowedUnaryOps: nil,
			ConstantMin: -10, ConstantMax: 10,
			VariableProbability: 0.5, TerminalProbability: 0.3, Seed: 0,
		}},
		{"TP=0", func() generator.Config {
			c := generator.Default(); c.TerminalProbability = 0; return c
		}()},
		{"TP=1", func() generator.Config {
			c := generator.Default(); c.TerminalProbability = 1; return c
		}()},
		{"ConstantMin=ConstantMax", func() generator.Config {
			c := generator.Default(); c.ConstantMin = 5; c.ConstantMax = 5; return c
		}()},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g, err := generator.New(tc.cfg)
			if err != nil {
				t.Errorf("error inesperado: %v", err)
			}
			if g == nil {
				t.Error("New devolvió nil sin error")
			}
		})
	}
}

func TestNew_Invalid(t *testing.T) {
	base := generator.Default()
	cases := []struct {
		name string
		cfg  generator.Config
	}{
		{"MaxDepth=0", func() generator.Config { c := base; c.MaxDepth = 0; return c }()},
		{"MaxDepth=-1", func() generator.Config { c := base; c.MaxDepth = -1; return c }()},
		{"MaxNodes=0", func() generator.Config { c := base; c.MaxNodes = 0; return c }()},
		{"MaxNodes=-1", func() generator.Config { c := base; c.MaxNodes = -1; return c }()},
		{"ConstantMin>Max", func() generator.Config {
			c := base; c.ConstantMin = 10; c.ConstantMax = 5; return c
		}()},
		{"TP=-0.1", func() generator.Config { c := base; c.TerminalProbability = -0.1; return c }()},
		{"TP=1.01", func() generator.Config { c := base; c.TerminalProbability = 1.01; return c }()},
		{"VP=-0.1", func() generator.Config { c := base; c.VariableProbability = -0.1; return c }()},
		{"VP=1.01", func() generator.Config { c := base; c.VariableProbability = 1.01; return c }()},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := generator.New(tc.cfg)
			if err == nil {
				t.Error("se esperaba error para config inválida, se obtuvo nil")
			}
		})
	}
}

// ─── Generación básica ────────────────────────────────────────────────────

func TestGenerate_NotNil(t *testing.T) {
	g, _ := generator.New(generator.Default())
	for i := 0; i < 20; i++ {
		if g.Generate() == nil {
			t.Errorf("árbol %d: Generate() devolvió nil", i)
		}
	}
}

func TestGenerate_GeneratesVariable(t *testing.T) {
	// VP=1, TP=1: cada árbol debe ser NodeVariable
	cfg := generator.Default()
	cfg.VariableProbability = 1
	cfg.TerminalProbability = 1
	g, _ := generator.New(cfg)
	for i := 0; i < 10; i++ {
		tree := g.Generate()
		if tree.Type != expression.NodeVariable {
			t.Errorf("árbol %d: Type=%v, want NodeVariable", i, tree.Type)
		}
	}
}

func TestGenerate_GeneratesConstant(t *testing.T) {
	// VP=0, TP=1: cada árbol debe ser NodeConstant
	cfg := generator.Default()
	cfg.VariableProbability = 0
	cfg.TerminalProbability = 1
	g, _ := generator.New(cfg)
	for i := 0; i < 10; i++ {
		tree := g.Generate()
		if tree.Type != expression.NodeConstant {
			t.Errorf("árbol %d: Type=%v, want NodeConstant", i, tree.Type)
		}
	}
}

// ─── Restricciones estructurales ──────────────────────────────────────────

func TestGenerate_RespectsMaxDepth(t *testing.T) {
	cfg := generator.Default()
	cfg.Seed = 1
	g, _ := generator.New(cfg)
	for i := 0; i < 200; i++ {
		tree := g.Generate()
		d := expression.Depth(tree)
		if d > cfg.MaxDepth {
			t.Errorf("árbol %d: Depth=%d > MaxDepth=%d", i, d, cfg.MaxDepth)
		}
	}
}

func TestGenerate_RespectsMaxNodes(t *testing.T) {
	cfg := generator.Default()
	cfg.Seed = 1
	g, _ := generator.New(cfg)
	for i := 0; i < 200; i++ {
		tree := g.Generate()
		n := expression.Count(tree)
		if n > cfg.MaxNodes {
			t.Errorf("árbol %d: Count=%d > MaxNodes=%d", i, n, cfg.MaxNodes)
		}
	}
}

func TestGenerate_LeavesAtMaxDepth(t *testing.T) {
	// TP=0: el generador siempre intenta crear operadores → crecimiento máximo.
	// Con MaxDepth=3 y solo NodeAdd, todos los árboles deben tener Depth=3
	// y las hojas deben estar exactamente en expression-depth 3.
	cfg := generator.Config{
		MaxDepth: 3, MaxNodes: 100,
		AllowedBinaryOps: []expression.NodeType{expression.NodeAdd},
		AllowedUnaryOps:  nil,
		ConstantMin: -10, ConstantMax: 10,
		VariableProbability: 0.5,
		TerminalProbability: 0, // siempre operadores (salvo forzado)
		Seed: 42,
	}
	g, _ := generator.New(cfg)
	for i := 0; i < 20; i++ {
		tree := g.Generate()
		if d := expression.Depth(tree); d != cfg.MaxDepth {
			t.Errorf("árbol %d: con TP=0 se esperaba Depth=%d, got %d", i, cfg.MaxDepth, d)
		}
		checkLeavesAtMaxDepth(t, tree, cfg.MaxDepth, 1)
	}
}

// ─── Estructura de nodos ──────────────────────────────────────────────────

func TestGenerate_BinaryStructure(t *testing.T) {
	g, _ := generator.New(generator.Default())
	binaryOps := map[expression.NodeType]bool{
		expression.NodeAdd: true, expression.NodeSub: true,
		expression.NodeMul: true, expression.NodeDiv: true,
	}
	for i := 0; i < 100; i++ {
		tree := g.Generate()
		traverseAll(tree, func(n *expression.Node) {
			if !binaryOps[n.Type] {
				return
			}
			if n.Left == nil {
				t.Errorf("árbol %d: nodo binario %v tiene Left=nil", i, n.Type)
			}
			if n.Right == nil {
				t.Errorf("árbol %d: nodo binario %v tiene Right=nil", i, n.Type)
			}
		})
	}
}

func TestGenerate_UnaryStructure(t *testing.T) {
	g, _ := generator.New(generator.Default())
	unaryOps := map[expression.NodeType]bool{
		expression.NodeSqrt: true, expression.NodeLn: true,
	}
	for i := 0; i < 100; i++ {
		tree := g.Generate()
		traverseAll(tree, func(n *expression.Node) {
			if !unaryOps[n.Type] {
				return
			}
			if n.Left == nil {
				t.Errorf("árbol %d: nodo unario %v tiene Left=nil", i, n.Type)
			}
			if n.Right != nil {
				t.Errorf("árbol %d: nodo unario %v tiene Right≠nil", i, n.Type)
			}
		})
	}
}

func TestGenerate_LeafStructure(t *testing.T) {
	g, _ := generator.New(generator.Default())
	for i := 0; i < 100; i++ {
		tree := g.Generate()
		traverseAll(tree, func(n *expression.Node) {
			if n.Type != expression.NodeVariable && n.Type != expression.NodeConstant {
				return
			}
			if n.Left != nil {
				t.Errorf("árbol %d: hoja %v tiene Left≠nil", i, n.Type)
			}
			if n.Right != nil {
				t.Errorf("árbol %d: hoja %v tiene Right≠nil", i, n.Type)
			}
		})
	}
}

// ─── Operadores y constantes ───────────────────────────────────────────────

func TestGenerate_OnlyAllowedOps(t *testing.T) {
	// Habilitar solo NodeAdd y NodeSqrt; el resto no debe aparecer.
	cfg := generator.Default()
	cfg.AllowedBinaryOps = []expression.NodeType{expression.NodeAdd}
	cfg.AllowedUnaryOps  = []expression.NodeType{expression.NodeSqrt}
	cfg.Seed = 77

	forbidden := map[expression.NodeType]bool{
		expression.NodeSub: true, expression.NodeMul: true,
		expression.NodeDiv: true, expression.NodeLn: true,
	}
	g, _ := generator.New(cfg)
	for i := 0; i < 100; i++ {
		traverseAll(g.Generate(), func(n *expression.Node) {
			if forbidden[n.Type] {
				t.Errorf("árbol %d: operador no habilitado %v", i, n.Type)
			}
		})
	}
}

func TestGenerate_ConstantsInRange(t *testing.T) {
	// VP=0: todas las hojas son constantes → verificar rango.
	cfg := generator.Default()
	cfg.ConstantMin         = 5
	cfg.ConstantMax         = 10
	cfg.VariableProbability = 0
	cfg.Seed = 55
	g, _ := generator.New(cfg)
	for i := 0; i < 100; i++ {
		traverseAll(g.Generate(), func(n *expression.Node) {
			if n.Type != expression.NodeConstant {
				return
			}
			if n.Value < cfg.ConstantMin || n.Value > cfg.ConstantMax {
				t.Errorf("árbol %d: constante %v fuera de [%v, %v]",
					i, n.Value, cfg.ConstantMin, cfg.ConstantMax)
			}
		})
	}
}

// ─── Reproducibilidad ─────────────────────────────────────────────────────

func TestGenerate_SeedReproducible(t *testing.T) {
	// Misma config + misma seed → misma secuencia de árboles.
	cfg := generator.Default()
	cfg.Seed = 42
	g1, _ := generator.New(cfg)
	g2, _ := generator.New(cfg)
	for i := 0; i < 10; i++ {
		s1 := expression.Format(g1.Generate())
		s2 := expression.Format(g2.Generate())
		if s1 != s2 {
			t.Errorf("árbol %d: misma seed produjo árboles diferentes:\n  %s\n  %s", i, s1, s2)
		}
	}
}

func TestGenerate_DifferentSeeds(t *testing.T) {
	// Seeds distintas deben producir al menos un árbol diferente en 10 intentos.
	cfg1 := generator.Default(); cfg1.Seed = 42
	cfg2 := generator.Default(); cfg2.Seed = 43
	g1, _ := generator.New(cfg1)
	g2, _ := generator.New(cfg2)

	hasDiff := false
	for i := 0; i < 10; i++ {
		if expression.Format(g1.Generate()) != expression.Format(g2.Generate()) {
			hasDiff = true
			break
		}
	}
	if !hasDiff {
		t.Error("10 pares de árboles idénticos con seeds distintas (estadísticamente improbable)")
	}
}

// ─── Casos borde ──────────────────────────────────────────────────────────

func TestGenerate_AllOpsDisabled(t *testing.T) {
	// Sin operadores: todos los árboles deben ser hojas únicas (Count=1, Depth=1).
	cfg := generator.Config{
		MaxDepth: 5, MaxNodes: 75,
		AllowedBinaryOps: nil, AllowedUnaryOps: nil,
		ConstantMin: -10, ConstantMax: 10,
		VariableProbability: 0.5, TerminalProbability: 0.3,
		Seed: 42,
	}
	g, _ := generator.New(cfg)
	for i := 0; i < 20; i++ {
		tree := g.Generate()
		if expression.Count(tree) != 1 {
			t.Errorf("árbol %d: Count=%d, want 1 (sin operadores)", i, expression.Count(tree))
		}
		if expression.Depth(tree) != 1 {
			t.Errorf("árbol %d: Depth=%d, want 1 (sin operadores)", i, expression.Depth(tree))
		}
	}
}

func TestGenerate_MaxDepth1(t *testing.T) {
	// MaxDepth=1: el generador solo puede producir hojas.
	cfg := generator.Default()
	cfg.MaxDepth = 1
	g, _ := generator.New(cfg)
	for i := 0; i < 10; i++ {
		tree := g.Generate()
		if expression.Depth(tree) != 1 {
			t.Errorf("árbol %d: MaxDepth=1 pero Depth=%d", i, expression.Depth(tree))
		}
		if expression.Count(tree) != 1 {
			t.Errorf("árbol %d: MaxDepth=1 pero Count=%d", i, expression.Count(tree))
		}
	}
}

func TestGenerate_Variety(t *testing.T) {
	// Con config default (TP=0.3, MaxDepth=7), 100 árboles deben mostrar
	// al menos 3 profundidades distintas y no todos deben ser hojas.
	cfg := generator.Default()
	cfg.Seed = 42
	g, _ := generator.New(cfg)

	depths := make(map[int]int)
	for i := 0; i < 100; i++ {
		d := expression.Depth(g.Generate())
		depths[d]++
	}
	if len(depths) < 3 {
		t.Errorf("se esperaban ≥ 3 profundidades distintas, se obtuvieron %d: %v",
			len(depths), depths)
	}
	if depths[1] == 100 {
		t.Error("todos los 100 árboles son hojas únicas (TP demasiado alto o error en grow)")
	}
}

func TestGenerate_IndependentTrees(t *testing.T) {
	// Modificar el clon de un árbol no debe afectar el original.
	g, _ := generator.New(generator.Default())
	for i := 0; i < 10; i++ {
		tree := g.Generate()
		clone := expression.Clone(tree)

		originalType := tree.Type
		clone.Type = expression.NodeConstant
		clone.Value = -9999
		if tree.Type != originalType {
			t.Errorf("árbol %d: modificar clone.Type afectó el original", i)
		}

		if clone.Left != nil && tree.Left != nil {
			originalLeftType := tree.Left.Type
			clone.Left = expression.NewConstant(12345)
			if tree.Left.Type != originalLeftType {
				t.Errorf("árbol %d: modificar clone.Left afectó original.Left", i)
			}
		}
	}
}

// ─── Tests adicionales de perfil FAST ────────────────────────────────────

func TestGenerate_FASTProfile(t *testing.T) {
	// Simular el perfil FAST (depth=5, nodes=40) y verificar restricciones.
	cfg := generator.Default()
	cfg.MaxDepth = 5
	cfg.MaxNodes = 40
	cfg.Seed = 7
	g, _ := generator.New(cfg)
	for i := 0; i < 100; i++ {
		tree := g.Generate()
		if d := expression.Depth(tree); d > 5 {
			t.Errorf("FAST árbol %d: Depth=%d > 5", i, d)
		}
		if n := expression.Count(tree); n > 40 {
			t.Errorf("FAST árbol %d: Count=%d > 40", i, n)
		}
	}
}
