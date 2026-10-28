package expression

import "fmt"

// Operator precedences for binary operators.
// Higher value = tighter binding.
const (
	precAddSub = 1
	precMulDiv = 2
	precPow    = 3
)

// precOf returns the precedence of a binary operator NodeType.
// Returns 0 for non-binary types (leaves, unary operators).
func precOf(t NodeType) int {
	switch t {
	case NodeAdd, NodeSub:
		return precAddSub
	case NodeMul, NodeDiv:
		return precMulDiv
	case NodePow:
		return precPow
	default:
		return 0
	}
}

// needsParens reports whether a binary child with the given myPrec requires
// parentheses when placed inside a parent operator of parentPrec at the
// given position (left or right child).
//
// Rules:
//   - Lower precedence child always needs parens.
//   - Same precedence, RIGHT child: parens for all operators except NodePow
//     (which is right-associative: x ^ 2 ^ 3 means x ^ (2 ^ 3)).
//   - Same precedence, LEFT child: parens ONLY for NodePow, to prevent
//     (x^2)^3 being written as x^2^3 which right-assoc would misread.
//   - Higher precedence child: no parens needed.
//
// Note on float64 non-associativity: Add(x, Add(1,2)) evaluates as x+(1+2).
// Displaying it as "x + 1 + 2" would imply left-to-right evaluation (x+1)+2,
// a different computation in float64. Hence right-children at equal precedence
// are always parenthesized (except NodePow, which is genuinely right-assoc).
func needsParens(myPrec, parentPrec int, isRightChild bool) bool {
	if myPrec < parentPrec {
		return true
	}
	if myPrec == parentPrec {
		if isRightChild {
			return parentPrec != precPow
		}
		return parentPrec == precPow
	}
	return false
}

// Format returns a human-readable infix representation of the expression tree
// with minimal parentheses, preserving the exact evaluation order of the tree.
//
// Operator precedences (lowest to highest): +/- < */÷ < ^
// Unary functions use f(arg) notation; the function's own parentheses act as
// grouping so no extra parens are added inside.
// A negative constant used as the base of NodePow is parenthesized to
// disambiguate from unary negation: (-3) ^ 2, not -3 ^ 2.
func Format(node *Node) string {
	return formatPrec(node, 0, false)
}

// formatPrec formats node in the context of a parent operator with the given
// parentPrec, positioned as left or right child (isRightChild).
// Binary nodes wrap themselves in parentheses when needsParens returns true.
func formatPrec(node *Node, parentPrec int, isRightChild bool) string {
	if node == nil {
		return "<nil>"
	}
	switch node.Type {
	case NodeVariable:
		return "x"
	case NodeConstant:
		// %g strips trailing zeros: 2.0 → "2", 3.14 → "3.14", -2.0 → "-2"
		return fmt.Sprintf("%g", node.Value)
	case NodeAdd, NodeSub, NodeMul, NodeDiv:
		return formatBinary(node, parentPrec, isRightChild)
	case NodePow:
		return formatPow(node, parentPrec, isRightChild)
	case NodeSqrt:
		return "sqrt(" + formatPrec(node.Left, 0, false) + ")"
	case NodeLn:
		return "ln(" + formatPrec(node.Left, 0, false) + ")"
	case NodeAbs:
		return "abs(" + formatPrec(node.Left, 0, false) + ")"
	case NodeExp:
		return "exp(" + formatPrec(node.Left, 0, false) + ")"
	case NodeSin:
		return "sin(" + formatPrec(node.Left, 0, false) + ")"
	case NodeCos:
		return "cos(" + formatPrec(node.Left, 0, false) + ")"
	default:
		return fmt.Sprintf("<node:%d>", node.Type)
	}
}

// formatBinary formats a binary operator node (Add, Sub, Mul, Div).
func formatBinary(node *Node, parentPrec int, isRightChild bool) string {
	myPrec := precOf(node.Type)
	var op string
	switch node.Type {
	case NodeAdd:
		op = "+"
	case NodeSub:
		op = "-"
	case NodeMul:
		op = "*"
	default: // NodeDiv
		op = "/"
	}
	leftStr := formatPrec(node.Left, myPrec, false)
	rightStr := formatPrec(node.Right, myPrec, true)
	inner := leftStr + " " + op + " " + rightStr
	if needsParens(myPrec, parentPrec, isRightChild) {
		return "(" + inner + ")"
	}
	return inner
}

// formatPow formats a NodePow node, applying a special case for negative
// constant bases to avoid ambiguity with unary minus:
// -3 ^ 2 could be read as -(3^2); (-3) ^ 2 is unambiguous.
func formatPow(node *Node, parentPrec int, isRightChild bool) string {
	leftStr := formatPrec(node.Left, precPow, false)
	if node.Left != nil && node.Left.Type == NodeConstant && node.Left.Value < 0 {
		leftStr = "(" + leftStr + ")"
	}
	rightStr := formatPrec(node.Right, precPow, true)
	inner := leftStr + " ^ " + rightStr
	if needsParens(precPow, parentPrec, isRightChild) {
		return "(" + inner + ")"
	}
	return inner
}
