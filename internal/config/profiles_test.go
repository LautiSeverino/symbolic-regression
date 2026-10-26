package config_test

import (
	"math"
	"testing"

	"github.com/LautiSeverino/symbolic-regression/internal/config"
	"github.com/LautiSeverino/symbolic-regression/internal/dataset"
	"github.com/LautiSeverino/symbolic-regression/internal/expression"
	"github.com/LautiSeverino/symbolic-regression/internal/fitness"
	"github.com/LautiSeverino/symbolic-regression/internal/genetic"
)

// ─── Helpers ──────────────────────────────────────────────────────────────

// minimalDS devuelve el dataset mínimo válido (2 puntos).
func minimalDS() *dataset.Dataset {
	return dataset.New([]dataset.Point{
		{X: 1, Y: 3},
		{X: 2, Y: 5},
	})
}

// linearDS devuelve el dataset canónico f(x) = 2*x + 1 con 5 puntos.
func linearDS() *dataset.Dataset {
	return dataset.New([]dataset.Point{
		{X: 1, Y: 3}, {X: 2, Y: 5}, {X: 3, Y: 7},
		{X: 4, Y: 9}, {X: 5, Y: 11},
	})
}

// allProfileNames devuelve los nombres de los cuatro perfiles.
var allProfileNames = []string{"fast", "balanced", "intensive", "benchmark"}

// budgetOf extrae los parámetros de presupuesto de una Config para comparación.
type budget struct {
	populationSize int
	generations    int
	workers        int
	maxDepth       int
	maxNodes       int
}

func budgetOf(cfg genetic.Config) budget {
	return budget{
		populationSize: cfg.PopulationSize,
		generations:    cfg.Generations,
		workers:        cfg.Workers,
		maxDepth:       cfg.GenConfig.MaxDepth,
		maxNodes:       cfg.GenConfig.MaxNodes,
	}
}

// ─── Parse ────────────────────────────────────────────────────────────────

func TestParse_ValidNames(t *testing.T) {
	for _, name := range allProfileNames {
		t.Run(name, func(t *testing.T) {
			_, err := config.Parse(name)
			if err != nil {
				t.Errorf("Parse(%q): error inesperado: %v", name, err)
			}
		})
	}
}

func TestParse_CaseInsensitive(t *testing.T) {
	cases := []string{"FAST", "Fast", "balanced", "BALANCED", "Intensive", "BENCHMARK"}
	for _, name := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := config.Parse(name)
			if err != nil {
				t.Errorf("Parse(%q): error inesperado con nombre en mayúsculas/minúsculas", name)
			}
		})
	}
}

func TestParse_TrimSpace(t *testing.T) {
	cases := []string{" fast", "balanced ", "  intensive  ", "\tbenchmark\t"}
	for _, name := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := config.Parse(name)
			if err != nil {
				t.Errorf("Parse(%q): error inesperado con espacios alrededor del nombre", name)
			}
		})
	}
}

func TestParse_InvalidName(t *testing.T) {
	invalid := []string{"ultra", "FAST2", "slow", "default", "normal", "x"}
	for _, name := range invalid {
		t.Run(name, func(t *testing.T) {
			_, err := config.Parse(name)
			if err == nil {
				t.Errorf("Parse(%q): se esperaba error para nombre desconocido, se obtuvo nil", name)
			}
		})
	}
}

func TestParse_EmptyString(t *testing.T) {
	_, err := config.Parse("")
	if err == nil {
		t.Error("Parse(\"\"): se esperaba error para cadena vacía, se obtuvo nil")
	}
}

// ─── Valores de presupuesto por perfil ────────────────────────────────────

func TestFASTConfig_BudgetValues(t *testing.T) {
	cfg := config.FASTConfig()
	want := budget{populationSize: 250, generations: 50, workers: 4, maxDepth: 5, maxNodes: 40}
	got := budgetOf(cfg)
	if got != want {
		t.Errorf("FAST budget = %+v, want %+v", got, want)
	}
}

func TestBALANCEDConfig_BudgetValues(t *testing.T) {
	cfg := config.BALANCEDConfig()
	want := budget{populationSize: 1_000, generations: 300, workers: 8, maxDepth: 7, maxNodes: 75}
	got := budgetOf(cfg)
	if got != want {
		t.Errorf("BALANCED budget = %+v, want %+v", got, want)
	}
}

func TestINTENSIVEConfig_BudgetValues(t *testing.T) {
	cfg := config.INTENSIVEConfig()
	want := budget{populationSize: 5_000, generations: 1_000, workers: 8, maxDepth: 10, maxNodes: 100}
	got := budgetOf(cfg)
	if got != want {
		t.Errorf("INTENSIVE budget = %+v, want %+v", got, want)
	}
}

func TestBENCHMARKConfig_MatchesBALANCED(t *testing.T) {
	// BENCHMARK y BALANCED deben tener exactamente los mismos parámetros de presupuesto.
	b := budgetOf(config.BALANCEDConfig())
	bm := budgetOf(config.BENCHMARKConfig())
	if b != bm {
		t.Errorf("BENCHMARK budget %+v ≠ BALANCED budget %+v", bm, b)
	}
}

func TestBENCHMARKConfig_MatchesBALANCED_AlgorithmParams(t *testing.T) {
	// Los parámetros del algoritmo también deben coincidir.
	b := config.BALANCEDConfig()
	bm := config.BENCHMARKConfig()
	if b.TournamentSize != bm.TournamentSize {
		t.Errorf("TournamentSize: BALANCED=%d, BENCHMARK=%d", b.TournamentSize, bm.TournamentSize)
	}
	if b.ElitismRate != bm.ElitismRate {
		t.Errorf("ElitismRate: BALANCED=%v, BENCHMARK=%v", b.ElitismRate, bm.ElitismRate)
	}
	if b.MutationRate != bm.MutationRate {
		t.Errorf("MutationRate: BALANCED=%v, BENCHMARK=%v", b.MutationRate, bm.MutationRate)
	}
	if b.CrossoverRate != bm.CrossoverRate {
		t.Errorf("CrossoverRate: BALANCED=%v, BENCHMARK=%v", b.CrossoverRate, bm.CrossoverRate)
	}
	if b.Lambda != bm.Lambda {
		t.Errorf("Lambda: BALANCED=%v, BENCHMARK=%v", b.Lambda, bm.Lambda)
	}
}

// ─── Seeds: ningún perfil las fija ────────────────────────────────────────

func TestAllProfiles_SeedIsZero(t *testing.T) {
	// Los perfiles no deben fijar seeds; el caller las asigna.
	for _, name := range allProfileNames {
		cfg, _ := config.Parse(name)
		if cfg.Seed != 0 {
			t.Errorf("%s: cfg.Seed = %d, want 0", name, cfg.Seed)
		}
		if cfg.GenConfig.Seed != 0 {
			t.Errorf("%s: cfg.GenConfig.Seed = %d, want 0", name, cfg.GenConfig.Seed)
		}
	}
}

// ─── Invariantes estructurales ────────────────────────────────────────────

func TestAllProfiles_WorkersAtLeastOne(t *testing.T) {
	for _, name := range allProfileNames {
		cfg, _ := config.Parse(name)
		if cfg.Workers < 1 {
			t.Errorf("%s: Workers = %d, want ≥ 1", name, cfg.Workers)
		}
	}
}

func TestAllProfiles_TournamentSizeLessThanPopulation(t *testing.T) {
	for _, name := range allProfileNames {
		cfg, _ := config.Parse(name)
		if cfg.TournamentSize >= cfg.PopulationSize {
			t.Errorf("%s: TournamentSize (%d) ≥ PopulationSize (%d)",
				name, cfg.TournamentSize, cfg.PopulationSize)
		}
	}
}

func TestAllProfiles_GenDepthMatchesBudget(t *testing.T) {
	// GenConfig.MaxDepth y MaxNodes deben coincidir con los valores de presupuesto.
	wantDepth := map[string]int{"fast": 5, "balanced": 7, "intensive": 10, "benchmark": 7}
	wantNodes := map[string]int{"fast": 40, "balanced": 75, "intensive": 100, "benchmark": 75}
	for _, name := range allProfileNames {
		cfg, _ := config.Parse(name)
		if cfg.GenConfig.MaxDepth != wantDepth[name] {
			t.Errorf("%s: GenConfig.MaxDepth = %d, want %d", name, cfg.GenConfig.MaxDepth, wantDepth[name])
		}
		if cfg.GenConfig.MaxNodes != wantNodes[name] {
			t.Errorf("%s: GenConfig.MaxNodes = %d, want %d", name, cfg.GenConfig.MaxNodes, wantNodes[name])
		}
	}
}

// ─── Parámetros del algoritmo: uniformes en todos los perfiles ─────────────

func TestAllProfiles_AlgorithmParamsUniform(t *testing.T) {
	// Los parámetros que no son de presupuesto deben ser idénticos en todos los perfiles.
	reference, _ := config.Parse("balanced")
	for _, name := range allProfileNames {
		cfg, _ := config.Parse(name)
		if cfg.TournamentSize != reference.TournamentSize {
			t.Errorf("%s: TournamentSize = %d, want %d", name, cfg.TournamentSize, reference.TournamentSize)
		}
		if cfg.ElitismRate != reference.ElitismRate {
			t.Errorf("%s: ElitismRate = %v, want %v", name, cfg.ElitismRate, reference.ElitismRate)
		}
		if cfg.MutationRate != reference.MutationRate {
			t.Errorf("%s: MutationRate = %v, want %v", name, cfg.MutationRate, reference.MutationRate)
		}
		if cfg.CrossoverRate != reference.CrossoverRate {
			t.Errorf("%s: CrossoverRate = %v, want %v", name, cfg.CrossoverRate, reference.CrossoverRate)
		}
		if cfg.Lambda != reference.Lambda {
			t.Errorf("%s: Lambda = %v, want %v", name, cfg.Lambda, reference.Lambda)
		}
		if cfg.GenConfig.ConstantMin != reference.GenConfig.ConstantMin {
			t.Errorf("%s: ConstantMin = %v, want %v", name, cfg.GenConfig.ConstantMin, reference.GenConfig.ConstantMin)
		}
		if cfg.GenConfig.ConstantMax != reference.GenConfig.ConstantMax {
			t.Errorf("%s: ConstantMax = %v, want %v", name, cfg.GenConfig.ConstantMax, reference.GenConfig.ConstantMax)
		}
		if cfg.GenConfig.VariableProbability != reference.GenConfig.VariableProbability {
			t.Errorf("%s: VariableProbability = %v, want %v", name, cfg.GenConfig.VariableProbability, reference.GenConfig.VariableProbability)
		}
		if cfg.GenConfig.TerminalProbability != reference.GenConfig.TerminalProbability {
			t.Errorf("%s: TerminalProbability = %v, want %v", name, cfg.GenConfig.TerminalProbability, reference.GenConfig.TerminalProbability)
		}
	}
}

// ─── Conjunto de operadores ────────────────────────────────────────────────

func TestAllProfiles_BinaryOps(t *testing.T) {
	// Deben estar: Add, Sub, Mul, Div. NodePow excluido.
	wantBinary := map[expression.NodeType]bool{
		expression.NodeAdd: true,
		expression.NodeSub: true,
		expression.NodeMul: true,
		expression.NodeDiv: true,
	}
	for _, name := range allProfileNames {
		cfg, _ := config.Parse(name)
		ops := cfg.GenConfig.AllowedBinaryOps
		if len(ops) != len(wantBinary) {
			t.Errorf("%s: len(AllowedBinaryOps) = %d, want %d", name, len(ops), len(wantBinary))
		}
		for _, op := range ops {
			if op == expression.NodePow {
				t.Errorf("%s: NodePow no debe estar en AllowedBinaryOps (es opt-in)", name)
			}
			if !wantBinary[op] {
				t.Errorf("%s: operador binario inesperado: %v", name, op)
			}
		}
	}
}

func TestAllProfiles_UnaryOps(t *testing.T) {
	// Deben estar: Sqrt, Ln, Abs, Exp, Sin, Cos (todos los de Fase 8).
	wantUnary := map[expression.NodeType]bool{
		expression.NodeSqrt: true,
		expression.NodeLn:   true,
		expression.NodeAbs:  true,
		expression.NodeExp:  true,
		expression.NodeSin:  true,
		expression.NodeCos:  true,
	}
	for _, name := range allProfileNames {
		cfg, _ := config.Parse(name)
		ops := cfg.GenConfig.AllowedUnaryOps
		if len(ops) != len(wantUnary) {
			t.Errorf("%s: len(AllowedUnaryOps) = %d, want %d", name, len(ops), len(wantUnary))
		}
		for _, op := range ops {
			if !wantUnary[op] {
				t.Errorf("%s: operador unario inesperado: %v", name, op)
			}
		}
	}
}

func TestAllProfiles_OperatorSetsAreIndependent(t *testing.T) {
	// Cada llamada a Parse debe devolver slices independientes (sin aliasing).
	cfg1, _ := config.Parse("fast")
	cfg2, _ := config.Parse("fast")
	if len(cfg1.GenConfig.AllowedBinaryOps) > 0 {
		cfg1.GenConfig.AllowedBinaryOps[0] = expression.NodePow
		if cfg2.GenConfig.AllowedBinaryOps[0] == expression.NodePow {
			t.Error("AllowedBinaryOps comparte el array subyacente entre llamadas (aliasing)")
		}
	}
}

// ─── Validez vía NewEvolution ──────────────────────────────────────────────

func TestAllProfiles_ValidViaNewEvolution(t *testing.T) {
	// Todos los perfiles deben producir configs que pasen la validación de NewEvolution.
	// Se usa el dataset mínimo (2 puntos); el objetivo es validar la config, no convergencia.
	ds := minimalDS()
	for _, name := range allProfileNames {
		t.Run(name, func(t *testing.T) {
			cfg, err := config.Parse(name)
			if err != nil {
				t.Fatalf("Parse: %v", err)
			}
			cfg.Seed = 42
			cfg.GenConfig.Seed = 43
			_, err = genetic.NewEvolution(cfg, ds, nil)
			if err != nil {
				t.Errorf("NewEvolution rechazó la config del perfil %q: %v", name, err)
			}
		})
	}
}

// ─── Determinismo ─────────────────────────────────────────────────────────

// TestAllProfiles_Determinism verifica que misma seed + mismo perfil = mismo resultado.
// Reduce el presupuesto para mantener el tiempo de test razonable, pero preserva
// los parámetros estructurales del perfil (MaxDepth, MaxNodes, operadores).
func TestAllProfiles_Determinism(t *testing.T) {
	if testing.Short() {
		t.Skip("test de determinismo omitido con -short")
	}
	ds := linearDS()

	for _, name := range allProfileNames {
		name := name
		t.Run(name, func(t *testing.T) {
			run := func() (string, float64) {
				cfg, err := config.Parse(name)
				if err != nil {
					t.Fatalf("Parse: %v", err)
				}
				// Presupuesto reducido: verificamos determinismo, no convergencia.
				cfg.PopulationSize = 30
				cfg.Generations = 5
				cfg.Seed = 42
				cfg.GenConfig.Seed = 43

				evol, err := genetic.NewEvolution(cfg, ds, nil)
				if err != nil {
					t.Fatalf("NewEvolution: %v", err)
				}
				best, err := evol.Run()
				if err != nil {
					t.Fatalf("Run: %v", err)
				}
				return expression.Format(best.Expr), best.Result.Fitness
			}

			expr1, fit1 := run()
			expr2, fit2 := run()

			if expr1 != expr2 {
				t.Errorf("perfil %s: misma seed → expresiones distintas:\n  %s\n  %s",
					name, expr1, expr2)
			}
			if fit1 != fit2 {
				t.Errorf("perfil %s: misma seed → fitness distintos: %v vs %v",
					name, fit1, fit2)
			}
		})
	}
}

// ─── Integración ──────────────────────────────────────────────────────────

// TestProfile_FAST_EndToEnd ejecuta el perfil FAST completo sobre el dataset
// lineal f(x) = 2*x + 1 y verifica que el resultado sea estructuralmente válido:
// expresión no nula, MSE y fitness finitos en training y validation.
//
// No comprueba convergencia: la especificación no define un umbral de error
// para el perfil FAST. El objetivo es detectar panics, errores de configuración
// y resultados no finitos, no medir la calidad de la búsqueda.
func TestProfile_FAST_EndToEnd(t *testing.T) {
	if testing.Short() {
		t.Skip("test de integración omitido con -short")
	}

	// Dataset sintético: f(x) = 2*x + 1, 20 puntos en [1, 20].
	ds := dataset.Generate(func(x float64) float64 { return 2*x + 1 }, 1, 20, 20)
	train, val, err := dataset.Split(ds, 0.80, 42)
	if err != nil {
		t.Fatalf("Split: %v", err)
	}

	cfg := config.FASTConfig()
	cfg.Seed = 42
	cfg.GenConfig.Seed = 43

	evol, err := genetic.NewEvolution(cfg, train, nil)
	if err != nil {
		t.Fatalf("NewEvolution: %v", err)
	}

	best, err := evol.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if best.Expr == nil {
		t.Fatal("la expresión resultante es nil")
	}
	if math.IsNaN(best.Result.Fitness) || math.IsInf(best.Result.Fitness, 0) {
		t.Errorf("training Fitness no es finito: %v", best.Result.Fitness)
	}
	if math.IsNaN(best.Result.MSE) || math.IsInf(best.Result.MSE, 0) {
		t.Errorf("training MSE no es finito: %v", best.Result.MSE)
	}

	valResult, err := fitness.Evaluate(best.Expr, val, 0)
	if err != nil {
		t.Fatalf("Evaluate en validation: %v", err)
	}
	if math.IsNaN(valResult.MSE) || math.IsInf(valResult.MSE, 0) {
		t.Errorf("validation MSE no es finito: %v", valResult.MSE)
	}

	t.Logf("Expression     : %s", expression.Format(best.Expr))
	t.Logf("Training MSE   : %v", best.Result.MSE)
	t.Logf("Validation MSE : %v", valResult.MSE)
	t.Logf("Complexity     : %d nodes", best.Result.Complexity)
}
