// gen_dataset generates the benchmark dataset:
//
//	benchmarks/datasets/benchmark_quadratic_625pts.csv
//
// f(x) = x^2 + 3x + 2 -- 625 uniformly distributed points on [1, 20].
// Run once from the repository root:
//
//	go run ./benchmarks/gen_dataset/main.go
//
// Commit the generated CSV so that the //go:embed directive in
// benchmarks/benchmark_test.go resolves at build time.
package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"

	"github.com/LautiSeverino/symbolic-regression/internal/dataset"
)

func main() {
	fn := func(x float64) float64 { return x*x + 3*x + 2 }
	ds := dataset.Generate(fn, 1.0, 20.0, 625)

	const outPath = "benchmarks/datasets/benchmark_quadratic_625pts.csv"

	if err := os.MkdirAll("benchmarks/datasets", 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "error creating directory: %v\n", err)
		os.Exit(1)
	}

	f, err := os.Create(outPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error creating file: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	w := csv.NewWriter(f)
	if err := w.Write([]string{"x", "y"}); err != nil {
		fmt.Fprintf(os.Stderr, "error writing header: %v\n", err)
		os.Exit(1)
	}

	for _, p := range ds.Points {
		row := []string{
			strconv.FormatFloat(p.X, 'g', -1, 64),
			strconv.FormatFloat(p.Y, 'g', -1, 64),
		}
		if err := w.Write(row); err != nil {
			fmt.Fprintf(os.Stderr, "error writing row: %v\n", err)
			os.Exit(1)
		}
	}
	w.Flush()
	if err := w.Error(); err != nil {
		fmt.Fprintf(os.Stderr, "error flushing: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Generated: %s (%d points)\n", outPath, ds.Len())
}
