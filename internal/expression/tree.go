package expression

// Depth devuelve la profundidad máxima del árbol.
// Una hoja tiene profundidad 1. nil devuelve 0.
func Depth(node *Node) int {
	if node == nil {
		return 0
	}
	l := Depth(node.Left)
	r := Depth(node.Right)
	if l > r {
		return l + 1
	}
	return r + 1
}

// Count devuelve el número total de nodos en el árbol.
func Count(node *Node) int {
	if node == nil {
		return 0
	}
	return 1 + Count(node.Left) + Count(node.Right)
}

// Clone devuelve una copia profunda del subárbol.
// Modificar el clon no afecta al original.
// Esta función es crítica para las operaciones de crossover y mutación.
func Clone(node *Node) *Node {
	if node == nil {
		return nil
	}
	return &Node{
		Type:  node.Type,
		Value: node.Value,
		Left:  Clone(node.Left),
		Right: Clone(node.Right),
	}
}
