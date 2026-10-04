package genetic

import "github.com/LautiSeverino/symbolic-regression/internal/expression"

// nodeRef representa un nodo con su contexto dentro del árbol clonado.
// Permite reemplazar el nodo modificando directamente a su padre.
type nodeRef struct {
	node   *expression.Node
	parent *expression.Node
	isLeft bool // true = hijo izquierdo del padre; false = hijo derecho
}

// collectRefs recorre root en preorden y devuelve todos los nodos con contexto.
// El nodo raíz tiene parent = nil.
func collectRefs(root *expression.Node) []nodeRef {
	var refs []nodeRef
	var walk func(n, parent *expression.Node, isLeft bool)
	walk = func(n, parent *expression.Node, isLeft bool) {
		if n == nil {
			return
		}
		refs = append(refs, nodeRef{n, parent, isLeft})
		walk(n.Left, n, true)
		walk(n.Right, n, false)
	}
	walk(root, nil, false)
	return refs
}

// Crossover realiza subtree crossover entre a y b.
//
// Garantías:
//   - a y b nunca se modifican
//   - el hijo no comparte nodos mutables con ninguno de los padres
//   - Depth(hijo) ≤ MaxDepth y Count(hijo) ≤ MaxNodes
//
// Retorna (hijo, true) si los límites se respetan.
// Retorna (nil, false) si el resultado los viola; el caller debe manejar el fallback.
func (e *Evolution) Crossover(a, b *expression.Node) (*expression.Node, bool) {
	// Clonar a — el hijo se construye sobre esta copia independiente
	child := expression.Clone(a)

	refsChild := collectRefs(child)
	refsB := collectRefs(b)
	if len(refsChild) == 0 || len(refsB) == 0 {
		return child, true
	}

	// Punto de corte en el hijo (clon de a) y subárbol donante de b
	tgt := refsChild[e.rng.Intn(len(refsChild))]
	src := refsB[e.rng.Intn(len(refsB))]
	donor := expression.Clone(src.node) // clon independiente del subárbol de b

	// Reemplazar el nodo apuntado en el hijo
	var result *expression.Node
	if tgt.parent == nil {
		// El punto de corte es la raíz: el donante es el nuevo árbol completo
		result = donor
	} else {
		if tgt.isLeft {
			tgt.parent.Left = donor
		} else {
			tgt.parent.Right = donor
		}
		result = child
	}

	// Validar límites
	maxD := e.cfg.GenConfig.MaxDepth
	maxN := e.cfg.GenConfig.MaxNodes
	if expression.Depth(result) > maxD || expression.Count(result) > maxN {
		return nil, false
	}
	return result, true
}
