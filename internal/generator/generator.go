// Package generator creates random expression trees for symbolic regression.
// It depends on internal/expression but expression does not depend on generator.
package generator

import (
	"fmt"
	"math/rand"

	"github.com/LautiSeverino/symbolic-regression/internal/expression"
)

// Config holds all parameters for random expression tree generation.
// Zero values are invalid; use Default() to get sensible defaults.
type Config struct {
	MaxDepth  int // maximum expression.Depth() of generated trees (≥ 1)
	MaxNodes  int // maximum expression.Count() of generated trees (≥ 1)

	AllowedBinaryOps []expression.NodeType // e.g. NodeAdd, NodeSub, NodeMul, NodeDiv
	AllowedUnaryOps  []expression.NodeType // e.g. NodeSqrt, NodeLn

	ConstantMin float64 // lower bound for generated constants (≤ ConstantMax)
	ConstantMax float64 // upper bound for generated constants

	// VariableProbability is P(leaf = Variable); ConstantProbability = 1 − VariableProbability.
	VariableProbability float64 // must be in [0, 1]

	// TerminalProbability controls the grow algorithm: at each non-forced position,
	// a terminal (leaf) is chosen with this probability instead of an operator.
	// Higher values → smaller trees; lower values → larger trees.
	// Typical range: 0.1–0.4.
	TerminalProbability float64 // must be in [0, 1]

	// PowExponentMin and PowExponentMax constrain the exponent generated when
	// NodePow is selected as operator. The generator always produces an integer
	// constant in [PowExponentMin, PowExponentMax] as the right child of NodePow.
	// These fields are ignored when NodePow is not in AllowedBinaryOps.
	// PowExponentMin must be ≤ PowExponentMax.
	// Note: crossover and mutation may produce NodePow with arbitrary exponents;
	// the evaluator handles those cases safely via EvalError / finiteOr.
	PowExponentMin int // e.g. -4
	PowExponentMax int // e.g.  4

	Seed int64 // RNG seed; same seed + same config → same sequence
}

// Default returns a Config matching the BALANCED profile (depth 7, nodes 75).
func Default() Config {
	return Config{
		MaxDepth: 7,
		MaxNodes: 75,
		AllowedBinaryOps: []expression.NodeType{
			expression.NodeAdd, expression.NodeSub,
			expression.NodeMul, expression.NodeDiv,
		},
		AllowedUnaryOps: []expression.NodeType{
			expression.NodeSqrt, expression.NodeLn,
		},
		ConstantMin:         -10,
		ConstantMax:         10,
		VariableProbability: 0.5,
		TerminalProbability: 0.3,
		PowExponentMin:      -4,
		PowExponentMax:      4,
		Seed:                0,
	}
}

// Generator produces random expression trees using a local seeded RNG.
// It is not safe for concurrent use without external synchronization.
type Generator struct {
	cfg Config
	rng *rand.Rand
}

// New validates cfg and returns a ready Generator.
// Returns an error for any invalid configuration parameter.
func New(cfg Config) (*Generator, error) {
	if cfg.MaxDepth < 1 {
		return nil, fmt.Errorf("generator: MaxDepth must be ≥ 1, got %d", cfg.MaxDepth)
	}
	if cfg.MaxNodes < 1 {
		return nil, fmt.Errorf("generator: MaxNodes must be ≥ 1, got %d", cfg.MaxNodes)
	}
	if cfg.ConstantMin > cfg.ConstantMax {
		return nil, fmt.Errorf("generator: ConstantMin (%v) > ConstantMax (%v)",
			cfg.ConstantMin, cfg.ConstantMax)
	}
	if cfg.TerminalProbability < 0 || cfg.TerminalProbability > 1 {
		return nil, fmt.Errorf("generator: TerminalProbability must be in [0, 1], got %v",
			cfg.TerminalProbability)
	}
	if cfg.VariableProbability < 0 || cfg.VariableProbability > 1 {
		return nil, fmt.Errorf("generator: VariableProbability must be in [0, 1], got %v",
			cfg.VariableProbability)
	}
	if cfg.PowExponentMin > cfg.PowExponentMax {
		return nil, fmt.Errorf("generator: PowExponentMin (%d) > PowExponentMax (%d)",
			cfg.PowExponentMin, cfg.PowExponentMax)
	}
	return &Generator{
		cfg: cfg,
		rng: rand.New(rand.NewSource(cfg.Seed)),
	}, nil
}

// Generate creates one random expression tree satisfying:
//
//	expression.Depth(tree) ≤ cfg.MaxDepth
//	expression.Count(tree) ≤ cfg.MaxNodes
//
// Strategy: "grow" algorithm. At each non-forced position a terminal is chosen
// with probability TerminalProbability, or an operator from the allowed sets.
// Both depth and node-budget constraints terminate recursion deterministically,
// so limits are always respected regardless of probability settings.
func (g *Generator) Generate() *expression.Node {
	return g.grow(0, g.cfg.MaxNodes)
}

// grow builds a subtree at go-depth `depth` using at most `budget` nodes.
//
// Go-depth to expression.Depth relationship:
//   root is go-depth 0 → expression.Depth contribution = 1
//   A leaf at go-depth d → expression.Depth of the whole tree ≥ d+1
//
// Force-leaf when go-depth ≥ MaxDepth−1 ensures the deepest possible leaf is
// at go-depth MaxDepth−1, giving expression.Depth ≤ MaxDepth.
//
// Force-leaf when budget ≤ 1 ensures expression.Count ≤ MaxNodes.
func (g *Generator) grow(depth, budget int) *expression.Node {
	canBinary := len(g.cfg.AllowedBinaryOps) > 0 && budget >= 3
	canUnary  := len(g.cfg.AllowedUnaryOps) > 0 && budget >= 2

	// Mandatory leaf: depth limit, budget exhausted, or no operator fits the budget
	if depth >= g.cfg.MaxDepth-1 || budget <= 1 || (!canBinary && !canUnary) {
		return g.randomLeaf()
	}

	// Grow: stochastic choice between terminal and operator
	if g.rng.Float64() < g.cfg.TerminalProbability {
		return g.randomLeaf()
	}

	// Choose operator type; binary preferred 2:1 when both are available
	switch {
	case canBinary && canUnary:
		if g.rng.Intn(3) < 2 {
			return g.buildBinary(depth, budget)
		}
		return g.buildUnary(depth, budget)
	case canBinary:
		return g.buildBinary(depth, budget)
	default:
		return g.buildUnary(depth, budget)
	}
}

// buildBinary generates a binary operator node and distributes the node budget
// between its children.
//
// NodePow special case: the right child is always a single integer constant in
// [PowExponentMin, PowExponentMax]. This keeps generated trees mathematically
// meaningful and avoids producing complex exponent subtrees that the evaluator
// would almost certainly reject as NaN or Inf. The left child (the base) gets
// budget−2 nodes (one consumed by NodePow itself, one reserved for the constant).
//
// All other binary operators: the left child receives half the child budget;
// the right child receives the actual remainder after the left subtree is built.
// This guarantees MaxNodes is never exceeded regardless of how the left subtree grows.
func (g *Generator) buildBinary(depth, budget int) *expression.Node {
	op := g.cfg.AllowedBinaryOps[g.rng.Intn(len(g.cfg.AllowedBinaryOps))]

	if op == expression.NodePow {
		// budget ≥ 3 (guaranteed by canBinary check in grow)
		// Layout: 1 (NodePow) + count(left) + 1 (constant) ≤ budget
		leftBudget := budget - 2
		if leftBudget < 1 {
			leftBudget = 1
		}
		left := g.grow(depth+1, leftBudget)

		rangeSize := g.cfg.PowExponentMax - g.cfg.PowExponentMin + 1
		if rangeSize < 1 {
			rangeSize = 1
		}
		exp := float64(g.cfg.PowExponentMin + g.rng.Intn(rangeSize))
		return expression.NewBinary(expression.NodePow, left, expression.NewConstant(exp))
	}

	childBudget := budget - 1 // this node consumes one slot

	leftBudget := childBudget / 2
	if leftBudget < 1 {
		leftBudget = 1
	}
	left := g.grow(depth+1, leftBudget)

	// Right child gets the actual remaining budget after left is built
	rightBudget := childBudget - expression.Count(left)
	if rightBudget < 1 {
		rightBudget = 1
	}
	right := g.grow(depth+1, rightBudget)

	return expression.NewBinary(op, left, right)
}

// buildUnary generates a unary operator node. The single child receives budget−1.
func (g *Generator) buildUnary(depth, budget int) *expression.Node {
	op := g.cfg.AllowedUnaryOps[g.rng.Intn(len(g.cfg.AllowedUnaryOps))]
	return expression.NewUnary(op, g.grow(depth+1, budget-1))
}

// randomLeaf returns a Variable or Constant node according to VariableProbability.
func (g *Generator) randomLeaf() *expression.Node {
	if g.rng.Float64() < g.cfg.VariableProbability {
		return expression.NewVariable()
	}
	v := g.cfg.ConstantMin + g.rng.Float64()*(g.cfg.ConstantMax-g.cfg.ConstantMin)
	return expression.NewConstant(v)
}
