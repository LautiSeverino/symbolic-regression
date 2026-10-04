package genetic

import "github.com/LautiSeverino/symbolic-regression/internal/expression"

// walkAll recorre node en preorden aplicando fn a cada nodo.
func walkAll(node *expression.Node, fn func(*expression.Node)) {
	if node == nil {
		return
	}
	fn(node)
	walkAll(node.Left, fn)
	walkAll(node.Right, fn)
}

// Mutate aplica una mutación aleatoria y retorna el árbol resultante.
// El original nunca se modifica: todas las mutaciones clonan antes de alterar.
// Tipos (elegidos uniformemente):
//
//	0 — cambiar valor de una constante
//	1 — cambiar tipo de una hoja (Variable ↔ Constant)
//	2 — cambiar operador respetando aridad
//	3 — reemplazar subárbol usando el generador
func (e *Evolution) Mutate(node *expression.Node) *expression.Node {
	switch e.rng.Intn(4) {
	case 0:
		return e.mutateConstant(node)
	case 1:
		return e.mutateLeaf(node)
	case 2:
		return e.mutateOperator(node)
	default:
		return e.mutateSubtree(node)
	}
}

// mutateConstant cambia el valor de una constante aleatoria al rango configurado.
func (e *Evolution) mutateConstant(node *expression.Node) *expression.Node {
	clone := expression.Clone(node)
	var consts []*expression.Node
	walkAll(clone, func(n *expression.Node) {
		if n.Type == expression.NodeConstant {
			consts = append(consts, n)
		}
	})
	if len(consts) == 0 {
		return clone // sin constantes: retornar clon sin cambios
	}
	t := consts[e.rng.Intn(len(consts))]
	t.Value = e.cfg.GenConfig.ConstantMin +
		e.rng.Float64()*(e.cfg.GenConfig.ConstantMax-e.cfg.GenConfig.ConstantMin)
	return clone
}

// mutateLeaf convierte una hoja aleatoria entre Variable y Constant.
func (e *Evolution) mutateLeaf(node *expression.Node) *expression.Node {
	clone := expression.Clone(node)
	var leaves []*expression.Node
	walkAll(clone, func(n *expression.Node) {
		if n.Type == expression.NodeVariable || n.Type == expression.NodeConstant {
			leaves = append(leaves, n)
		}
	})
	if len(leaves) == 0 {
		return clone
	}
	t := leaves[e.rng.Intn(len(leaves))]
	if e.rng.Float64() < e.cfg.GenConfig.VariableProbability {
		t.Type = expression.NodeVariable
		t.Value = 0
	} else {
		t.Type = expression.NodeConstant
		t.Value = e.cfg.GenConfig.ConstantMin +
			e.rng.Float64()*(e.cfg.GenConfig.ConstantMax-e.cfg.GenConfig.ConstantMin)
	}
	t.Left = nil
	t.Right = nil
	return clone
}

// mutateOperator reemplaza un operador por otro del mismo aridad.
// Si solo hay un operador de ese aridad disponible, el árbol queda sin cambios.
func (e *Evolution) mutateOperator(node *expression.Node) *expression.Node {
	clone := expression.Clone(node)
	var ops []*expression.Node
	walkAll(clone, func(n *expression.Node) {
		if e.isBinaryAllowed(n.Type) || e.isUnaryAllowed(n.Type) {
			ops = append(ops, n)
		}
	})
	if len(ops) == 0 {
		return clone
	}
	t := ops[e.rng.Intn(len(ops))]

	if e.isBinaryAllowed(t.Type) {
		// Filtrar para no elegir el mismo operador
		others := make([]expression.NodeType, 0, len(e.cfg.GenConfig.AllowedBinaryOps))
		for _, op := range e.cfg.GenConfig.AllowedBinaryOps {
			if op != t.Type {
				others = append(others, op)
			}
		}
		if len(others) > 0 {
			t.Type = others[e.rng.Intn(len(others))]
		}
	} else {
		others := make([]expression.NodeType, 0, len(e.cfg.GenConfig.AllowedUnaryOps))
		for _, op := range e.cfg.GenConfig.AllowedUnaryOps {
			if op != t.Type {
				others = append(others, op)
			}
		}
		if len(others) > 0 {
			t.Type = others[e.rng.Intn(len(others))]
		}
	}
	return clone
}

// mutateSubtree reemplaza el subárbol de un nodo aleatorio con uno generado.
// Si el resultado viola MaxDepth o MaxNodes, retorna el clon original sin cambios.
func (e *Evolution) mutateSubtree(node *expression.Node) *expression.Node {
	clone := expression.Clone(node)
	refs := collectRefs(clone)
	if len(refs) == 0 {
		return clone
	}
	tgt := refs[e.rng.Intn(len(refs))]
	newSub := e.gen.Generate() // árbol independiente, respeta GenConfig limits

	var result *expression.Node
	if tgt.parent == nil {
		result = newSub
	} else {
		if tgt.isLeft {
			tgt.parent.Left = newSub
		} else {
			tgt.parent.Right = newSub
		}
		result = clone
	}

	maxD := e.cfg.GenConfig.MaxDepth
	maxN := e.cfg.GenConfig.MaxNodes
	if expression.Depth(result) <= maxD && expression.Count(result) <= maxN {
		return result
	}
	// Fallback: retornar árbol original sin mutación de subárbol
	return expression.Clone(node)
}

// isBinaryAllowed reporta si t está en AllowedBinaryOps.
func (e *Evolution) isBinaryAllowed(t expression.NodeType) bool {
	for _, op := range e.cfg.GenConfig.AllowedBinaryOps {
		if op == t {
			return true
		}
	}
	return false
}

// isUnaryAllowed reporta si t está en AllowedUnaryOps.
func (e *Evolution) isUnaryAllowed(t expression.NodeType) bool {
	for _, op := range e.cfg.GenConfig.AllowedUnaryOps {
		if op == t {
			return true
		}
	}
	return false
}
