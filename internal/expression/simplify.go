package expression

import "math"

// Simplify returns a new expression tree that is algebraically equivalent
// to node under the evaluator's semantics, with basic identity and zero
// element reductions applied bottom-up.
//
// Guarantees:
//   - Simplify(nil) == nil
//   - The original tree is never modified; the result is an independent copy.
//   - Simplify(Simplify(t)) is structurally identical to Simplify(t) (idempotent).
//   - EvalError behavior is preserved: a rule is only applied when doing so
//     cannot hide an error that the original expression would have produced.
//
// Rules applied:
//
//	x + 0  → x   (identity for addition)
//	x - 0  → x   (subtracting zero leaves x unchanged)
//	x * 1  → x   (identity for multiplication)
//	x / 1  → x   (dividing by one leaves x unchanged)
//	0 / c  → 0   (zero numerator; c must be a nonzero finite constant)
//	x * 0  → 0   (zero factor on right; x must be a safe leaf)
//	0 * x  → 0   (zero factor on left;  x must be a safe leaf)
//
// A "safe leaf" is a NodeVariable or a NodeConstant with a finite value —
// either is guaranteed never to produce EvalError. A compound subtree is
// conservatively treated as unsafe because it may contain a division, sqrt,
// ln, or other operation that can fail for some inputs. Simplifying x*0 → 0
// when x is unsafe would hide those errors.
//
// The symmetric forms 0+x and 1*x are intentionally omitted (out of scope).
// NodePow has no simplification rules in this phase.
func Simplify(node *Node) *Node {
	if node == nil {
		return nil
	}

	// Phase 1 — bottom-up: simplify children before examining this node.
	left := Simplify(node.Left)
	right := Simplify(node.Right)

	// Phase 2 — reconstruct with simplified children.
	// Using a fresh node ensures the original tree is never modified.
	result := &Node{
		Type:  node.Type,
		Value: node.Value,
		Left:  left,
		Right: right,
	}

	// Phase 3 — apply reduction rules at this level.
	switch result.Type {

	case NodeAdd:
		// x + 0 → x
		if isZeroConst(result.Right) {
			return result.Left
		}

	case NodeSub:
		// x - 0 → x
		if isZeroConst(result.Right) {
			return result.Left
		}

	case NodeMul:
		// x * 1 → x
		if isOneConst(result.Right) {
			return result.Left
		}
		// x * 0 → 0  (only when x is a safe leaf; otherwise preserve EvalError)
		if isZeroConst(result.Right) && isSafeLeaf(result.Left) {
			return &Node{Type: NodeConstant, Value: 0}
		}
		// 0 * x → 0  (symmetric; same safe-leaf constraint on x)
		if isZeroConst(result.Left) && isSafeLeaf(result.Right) {
			return &Node{Type: NodeConstant, Value: 0}
		}

	case NodeDiv:
		// x / 1 → x
		if isOneConst(result.Right) {
			return result.Left
		}
		// 0 / c → 0  (only when c is a nonzero finite constant)
		if isZeroConst(result.Left) && isNonzeroFiniteConst(result.Right) {
			return &Node{Type: NodeConstant, Value: 0}
		}
	}

	return result
}

// isZeroConst reports whether node is a NodeConstant with value exactly 0.
func isZeroConst(node *Node) bool {
	return node != nil && node.Type == NodeConstant && node.Value == 0
}

// isOneConst reports whether node is a NodeConstant with value exactly 1.
func isOneConst(node *Node) bool {
	return node != nil && node.Type == NodeConstant && node.Value == 1
}

// isSafeLeaf reports whether node is guaranteed never to produce EvalError
// for any finite input x:
//   - NodeVariable always returns x without error.
//   - NodeConstant with a finite value: Evaluate rejects NaN/±Inf immediately
//     (Phase 9), so a finite constant is always safe.
//
// Any compound subtree is conservatively treated as unsafe.
func isSafeLeaf(node *Node) bool {
	if node == nil {
		return false
	}
	switch node.Type {
	case NodeVariable:
		return true
	case NodeConstant:
		return !math.IsNaN(node.Value) && !math.IsInf(node.Value, 0)
	default:
		return false
	}
}

// isNonzeroFiniteConst reports whether node is a NodeConstant with a nonzero,
// finite value. Used to validate the denominator in 0/c → 0.
func isNonzeroFiniteConst(node *Node) bool {
	if node == nil || node.Type != NodeConstant {
		return false
	}
	v := node.Value
	return v != 0 && !math.IsNaN(v) && !math.IsInf(v, 0)
}
