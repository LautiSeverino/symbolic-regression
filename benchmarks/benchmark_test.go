package benchmark_test

import (
	"bytes"
	_ "embed"
	"testing"
	"time"

	benchconfig "github.com/LautiSeverino/symbolic-regression/internal/config"
	"github.com/LautiSeverino/symbolic-regression/internal/dataset"
	"github.com/LautiSeverino/symbolic-regression/internal/genetic"
)

//go:embed datasets/benchmark_quadratic_625pts.csv
var benchmarkDatasetCSV []byte

const (
	benchSeed    int64 = 42
	benchGenSeed int64 = 43 // = benchSeed + 1, same convention as main.go

	// totalPointEvals is the exact number of dataset-point evaluations per Run():
	//   InitPopulation:        1000 * 500         =   500 000
	//   NextGeneration (x100): 100 * 950 * 500    = 47 500 000
	//   finalize (val):        1 * 125             =       125
	//   Total:                                     = 48 000 125
	//
	// EliteCount = int(0.05 * 1000) = 50. Elites are copied without re-evaluation.
	// finalize() is called exactly once at the end of Run().
	totalPointEvals = 48_000_125
)

// buildBenchConfig returns the genetic.Config used in all benchmark iterations.
// BENCHMARKConfig() is the single source of truth; only four experiment-specific
// values are overridden.
func buildBenchConfig(workers int) genetic.Config {
	cfg := benchconfig.BENCHMARKConfig()
	cfg.Seed = benchSeed
	cfg.GenConfig.Seed = benchGenSeed
	cfg.Workers = workers
	cfg.Generations = 100 // overrides the 300 in BENCHMARKConfig
	// TargetMSEEnabled=false and StagnationLimit=0 are already in BENCHMARKConfig.
	return cfg
}

// benchmarkRun is the core benchmark function.
// Both ns/op (Go framework timer) and Meval/s (manual timer) measure
// exclusively evol.Run(). NewEvolution is excluded from both timers.
func benchmarkRun(b *testing.B, workers int) {
	b.Helper()

	ds, err := dataset.ParseCSV(bytes.NewReader(benchmarkDatasetCSV))
	if err != nil {
		b.Fatalf("dataset: %v", err)
	}
	train, val, err := dataset.Split(ds, 0.80, benchSeed)
	if err != nil {
		b.Fatalf("split: %v", err)
	}
	cfg := buildBenchConfig(workers)

	var totalElapsed time.Duration

	b.ResetTimer()
	b.StopTimer() // setup and result handling stay outside the benchmark timer
	for i := 0; i < b.N; i++ {
		evol, err := genetic.NewEvolution(cfg, train, val)
		if err != nil {
			b.Fatalf("NewEvolution: %v", err)
		}

		// Run() is the sole measured operation. Stop the framework timer
		// immediately afterward, before timing or handling the result.
		iterStart := time.Now()
		b.StartTimer()
		best, err := evol.Run()
		b.StopTimer()
		totalElapsed += time.Since(iterStart)
		if err != nil {
			b.Fatalf("Run: %v", err)
		}
		_ = best
	}

	if b.N > 0 {
		avgElapsed := totalElapsed / time.Duration(b.N)
		b.ReportMetric(float64(totalPointEvals)/avgElapsed.Seconds()/1e6, "Meval/s")
	}
}

// ── Formal benchmark set: Workers = 1, 2, 4, 8 ───────────────────────────

func BenchmarkRun_Workers1(b *testing.B) { benchmarkRun(b, 1) }
func BenchmarkRun_Workers2(b *testing.B) { benchmarkRun(b, 2) }
func BenchmarkRun_Workers4(b *testing.B) { benchmarkRun(b, 4) }
func BenchmarkRun_Workers8(b *testing.B) { benchmarkRun(b, 8) }

// ── Complementary experiment: Workers = 12 ────────────────────────────────
// Specific to the i7-1255U (12 logical CPUs). NOT part of the formal set.
// Run separately after the formal set is complete:
//
//	go test -bench=BenchmarkRun_Workers12 -benchtime=1x -count=5 ./benchmarks/
func BenchmarkRun_Workers12(b *testing.B) { benchmarkRun(b, 12) }
