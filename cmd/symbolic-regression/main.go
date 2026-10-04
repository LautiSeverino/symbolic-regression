package main

import (
	"fmt"

	"github.com/LautiSeverino/symbolic-regression/internal/expression"
)

func main() {
	// Fase 1: smoke test con el ejemplo canónico f(x) = 2*x + 1
	tree := expression.NewBinary(expression.NodeAdd,
		expression.NewBinary(expression.NodeMul,
			expression.NewConstant(2),
			expression.NewVariable(),
		),
		expression.NewConstant(1),
	)

	fmt.Printf("Expression : %s\n", expression.Format(tree))
	fmt.Printf("Depth      : %d\n", expression.Depth(tree))
	fmt.Printf("Nodes      : %d\n", expression.Count(tree))
	fmt.Println()
	fmt.Printf("%-5s  %s\n", "x", "f(x)")
	fmt.Println("─────────────")
	for x := 1.0; x <= 5; x++ {
		y, err := expression.Evaluate(tree, x)
		if err != nil {
			fmt.Printf("%-5.0f  ERR: %v\n", x, err)
			continue
		}
		fmt.Printf("%-5.0f  %.0f\n", x, y)
	}
}
