package expression

import "fmt"

// Format devuelve una representación infix legible del árbol de expresión.
//
// Estrategia actual: todo subárbol binario se parentesiza.
// Esto es siempre correcto aunque produce paréntesis redundantes.
// El formateo con precedencia mínima de paréntesis se implementa en la Fase 13.
func Format(node *Node) string {
	if node == nil {
		return "<nil>"
	}
	switch node.Type {
	case NodeVariable:
		return "x"
	case NodeConstant:
		// %g elimina ceros finales: 2.0 → "2", 3.14 → "3.14", -2.0 → "-2"
		return fmt.Sprintf("%g", node.Value)
	case NodeAdd:
		return fmt.Sprintf("(%s + %s)", Format(node.Left), Format(node.Right))
	case NodeSub:
		return fmt.Sprintf("(%s - %s)", Format(node.Left), Format(node.Right))
	case NodeMul:
		return fmt.Sprintf("(%s * %s)", Format(node.Left), Format(node.Right))
	case NodeDiv:
		return fmt.Sprintf("(%s / %s)", Format(node.Left), Format(node.Right))
	case NodeSqrt:
		return fmt.Sprintf("sqrt(%s)", Format(node.Left))
	case NodeLn:
		return fmt.Sprintf("ln(%s)", Format(node.Left))
	default:
		return fmt.Sprintf("<node:%d>", node.Type)
	}
}
