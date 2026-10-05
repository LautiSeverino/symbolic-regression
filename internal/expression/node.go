package expression

// NodeType identifica el tipo de un nodo en el árbol de expresión.
type NodeType int

const (
	NodeVariable NodeType = iota // la variable de entrada x
	NodeConstant                 // literal float64
	NodeAdd                      // binario: left + right
	NodeSub                      // binario: left - right
	NodeMul                      // binario: left * right
	NodeDiv                      // binario: left / right
	NodeSqrt                     // unario: sqrt(left)  — Fase 2
	NodeLn                       // unario: ln(left)    — Fase 2
	NodePow                      // binario: left ^ right — Fase 8
	NodeAbs                      // unario: abs(left)    — Fase 8
	NodeExp                      // unario: exp(left)    — Fase 8
	NodeSin                      // unario: sin(left)    — Fase 8
	NodeCos                      // unario: cos(left)    — Fase 8
)

// Node es un elemento del árbol de expresión.
//
// Operadores binarios (Add, Sub, Mul, Div) usan Left y Right.
// Funciones unarias (Sqrt, Ln, ...) usan Left únicamente; Right queda en nil.
// Hojas (Variable, Constant) tienen Left y Right en nil.
type Node struct {
	Type  NodeType
	Value float64 // significativo solo cuando Type == NodeConstant
	Left  *Node
	Right *Node
}

// NewVariable devuelve una hoja que representa la variable x.
func NewVariable() *Node {
	return &Node{Type: NodeVariable}
}

// NewConstant devuelve una hoja que representa un literal float64.
func NewConstant(v float64) *Node {
	return &Node{Type: NodeConstant, Value: v}
}

// NewBinary devuelve un nodo interno para un operador binario.
// t debe ser NodeAdd, NodeSub, NodeMul o NodeDiv.
func NewBinary(t NodeType, left, right *Node) *Node {
	return &Node{Type: t, Left: left, Right: right}
}

// NewUnary devuelve un nodo interno para una función unaria.
// El argumento de la función va en Left; Right queda en nil.
// t debe ser NodeSqrt, NodeLn u otro NodeType unario.
func NewUnary(t NodeType, arg *Node) *Node {
	return &Node{Type: t, Left: arg}
}
