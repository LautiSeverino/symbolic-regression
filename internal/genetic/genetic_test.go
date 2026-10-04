package genetic_test

import (
	"math"
	"sort"
	"testing"

	"github.com/LautiSeverino/symbolic-regression/internal/dataset"
	"github.com/LautiSeverino/symbolic-regression/internal/expression"
	"github.com/LautiSeverino/symbolic-regression/internal/fitness"
	"github.com/LautiSeverino/symbolic-regression/internal/genetic"
	"github.com/LautiSeverino/symbolic-regression/internal/generator"
)

// ─── Helpers ──────────────────────────────────────────────────────────────

// testCfg devuelve una configuración pequeña para tests rápidos.
func testCfg() genetic.Config {
	return genetic.Config{
		PopulationSize: 20,
		Generations:    5,
		TournamentSize: 3,
		ElitismRate:    0.1,
		MutationRate:   0.2,
		CrossoverRate:  0.8,
		Lambda:         0.01,
		Seed:           42,
		GenConfig: generator.Config{
			MaxDepth:            4,
			MaxNodes:            15,
			AllowedBinaryOps:    []expression.NodeType{expression.NodeAdd, expression.NodeMul},
			AllowedUnaryOps:     nil,
			ConstantMin:         -5,
			ConstantMax:         5,
			VariableProbability: 0.5,
			TerminalProbability: 0.3,
			Seed:                42,
		},
	}
}

// testDS devuelve el dataset canónico y = 2*x + 1.
func testDS() *dataset.Dataset {
	return dataset.New([]dataset.Point{
		{X: 1, Y: 3}, {X: 2, Y: 5}, {X: 3, Y: 7}, {X: 4, Y: 9}, {X: 5, Y: 11},
	})
}

// walkAllTest recorre el árbol aplicando fn a cada nodo (helper de test).
func walkAllTest(node *expression.Node, fn func(*expression.Node)) {
	if node == nil {
		return
	}
	fn(node)
	walkAllTest(node.Left, fn)
	walkAllTest(node.Right, fn)
}

// ─── NewEvolution ─────────────────────────────────────────────────────────

func TestNewEvolution_Valid(t *testing.T) {
	evol, err := genetic.NewEvolution(testCfg(), testDS())
	if err != nil {
		t.Fatalf("error inesperado: %v", err)
	}
	if evol == nil {
		t.Error("NewEvolution devolvió nil sin error")
	}
}

func TestNewEvolution_Invalid(t *testing.T) {
	base := testCfg()
	cases := []struct {
		name string
		cfg  genetic.Config
	}{
		{"PopulationSize<2", func() genetic.Config { c := base; c.PopulationSize = 1; return c }()},
		{"Generations<1", func() genetic.Config { c := base; c.Generations = 0; return c }()},
		{"TournamentSize<2", func() genetic.Config { c := base; c.TournamentSize = 1; return c }()},
		{"TournamentSize>Pop", func() genetic.Config {
			c := base; c.TournamentSize = base.PopulationSize + 1; return c
		}()},
		{"ElitismRate>=1", func() genetic.Config { c := base; c.ElitismRate = 1.0; return c }()},
		{"MutationRate>1", func() genetic.Config { c := base; c.MutationRate = 1.1; return c }()},
		{"DatasetNil", base},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var ds *dataset.Dataset
			if tc.name != "DatasetNil" {
				ds = testDS()
			}
			_, err := genetic.NewEvolution(tc.cfg, ds)
			if err == nil {
				t.Error("se esperaba error para config/dataset inválido, se obtuvo nil")
			}
		})
	}
}

// ─── Population ───────────────────────────────────────────────────────────

func TestInitPopulation_Size(t *testing.T) {
	cfg := testCfg()
	evol, _ := genetic.NewEvolution(cfg, testDS())
	pop := evol.InitPopulation()
	if len(pop) != cfg.PopulationSize {
		t.Errorf("len(pop) = %d, want %d", len(pop), cfg.PopulationSize)
	}
}

func TestInitPopulation_ValidExpr(t *testing.T) {
	evol, _ := genetic.NewEvolution(testCfg(), testDS())
	for i, ind := range evol.InitPopulation() {
		if ind.Expr == nil {
			t.Errorf("individuo %d: Expr es nil", i)
		}
	}
}

func TestInitPopulation_FitnessComputed(t *testing.T) {
	evol, _ := genetic.NewEvolution(testCfg(), testDS())
	for i, ind := range evol.InitPopulation() {
		if math.IsNaN(ind.Result.Fitness) || math.IsInf(ind.Result.Fitness, 0) {
			t.Errorf("individuo %d: Fitness no finito: %v", i, ind.Result.Fitness)
		}
	}
}

// ─── Tournament Selection ─────────────────────────────────────────────────

func TestTournamentSelect_FromPop(t *testing.T) {
	evol, _ := genetic.NewEvolution(testCfg(), testDS())
	pop := evol.InitPopulation()

	for i := 0; i < 30; i++ {
		selected := evol.TournamentSelect(pop)
		found := false
		for _, ind := range pop {
			if ind.Expr == selected.Expr {
				found = true
				break
			}
		}
		if !found {
			t.Fatal("TournamentSelect devolvió un individuo que no está en la población")
		}
	}
}

func TestTournamentSelect_SelectsBest(t *testing.T) {
	evol, _ := genetic.NewEvolution(testCfg(), testDS())

	// Crear pop de 10 con un ganador claro (fitness = 0)
	pop := make([]genetic.Individual, 10)
	for i := range pop {
		pop[i] = genetic.Individual{
			Expr:   expression.NewVariable(),
			Result: fitness.FitnessResult{Fitness: 9999},
		}
	}
	pop[0].Result.Fitness = 0 // ganador claro

	// P(ganador no aparece en 50 torneos) es ~0 con 10 candidatos y size=3
	bestSeen := false
	for i := 0; i < 50; i++ {
		if evol.TournamentSelect(pop).Result.Fitness == 0 {
			bestSeen = true
			break
		}
	}
	if !bestSeen {
		t.Error("el mejor individuo nunca fue seleccionado en 50 torneos (improbable si el torneo funciona)")
	}
}

// ─── Elitism ──────────────────────────────────────────────────────────────

func TestElitism_BestSurvive(t *testing.T) {
	cfg := testCfg()
	evol, _ := genetic.NewEvolution(cfg, testDS())

	pop := evol.InitPopulation()
	sort.Slice(pop, func(i, j int) bool {
		return pop[i].Result.Fitness < pop[j].Result.Fitness
	})
	bestFormat := expression.Format(pop[0].Expr)
	bestFitness := pop[0].Result.Fitness

	next := evol.NextGeneration(pop)

	found := false
	for _, ind := range next {
		if expression.Format(ind.Expr) == bestFormat && ind.Result.Fitness == bestFitness {
			found = true
			break
		}
	}
	if !found {
		t.Error("el mejor individuo no sobrevivió por elitismo a la siguiente generación")
	}
}

func TestElitism_Count(t *testing.T) {
	cfg := testCfg()
	cfg.ElitismRate = 0.1 // 10% de 20 = 2 élites
	evol, _ := genetic.NewEvolution(cfg, testDS())

	pop := evol.InitPopulation()
	sort.Slice(pop, func(i, j int) bool {
		return pop[i].Result.Fitness < pop[j].Result.Fitness
	})

	// Los 2 mejores deben aparecer en la siguiente generación
	eliteFormats := []string{
		expression.Format(pop[0].Expr),
		expression.Format(pop[1].Expr),
	}

	next := evol.NextGeneration(pop)
	for _, ef := range eliteFormats {
		found := false
		for _, ind := range next {
			if expression.Format(ind.Expr) == ef {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("élite con formato %q no encontrado en la siguiente generación", ef)
		}
	}
}

// ─── Crossover ────────────────────────────────────────────────────────────

// crossoverCfg devuelve una config con límites más holgados para reducir rechazos.
func crossoverCfg() genetic.Config {
	c := testCfg()
	c.GenConfig.MaxDepth = 7
	c.GenConfig.MaxNodes = 50
	return c
}

func TestCrossover_ReturnsValidTree(t *testing.T) {
	evol, _ := genetic.NewEvolution(crossoverCfg(), testDS())
	a := expression.NewBinary(expression.NodeAdd, expression.NewVariable(), expression.NewConstant(1))
	b := expression.NewBinary(expression.NodeMul, expression.NewVariable(), expression.NewConstant(2))

	attempts, success := 0, 0
	for attempts < 30 && success < 5 {
		if child, ok := evol.Crossover(a, b); ok {
			if child == nil {
				t.Fatal("crossover ok pero child es nil")
			}
			success++
		}
		attempts++
	}
	if success == 0 {
		t.Error("crossover rechazó todos los intentos con árboles pequeños (inesperado)")
	}
}

func TestCrossover_RespectsMaxDepth(t *testing.T) {
	cfg := crossoverCfg()
	evol, _ := genetic.NewEvolution(cfg, testDS())
	evol2, _ := genetic.NewEvolution(crossoverCfg(), testDS())

	pop1 := evol.InitPopulation()
	pop2 := evol2.InitPopulation()

	rejected := 0
	for i := 0; i < 50; i++ {
		a := pop1[i%len(pop1)].Expr
		b := pop2[i%len(pop2)].Expr
		if child, ok := evol.Crossover(a, b); ok {
			if d := expression.Depth(child); d > cfg.GenConfig.MaxDepth {
				t.Errorf("crossover %d: Depth=%d > MaxDepth=%d", i, d, cfg.GenConfig.MaxDepth)
			}
		} else {
			rejected++
		}
	}
	_ = rejected // algunos rechazos son esperables
}

func TestCrossover_RespectsMaxNodes(t *testing.T) {
	cfg := crossoverCfg()
	evol, _ := genetic.NewEvolution(cfg, testDS())
	evol2, _ := genetic.NewEvolution(crossoverCfg(), testDS())

	pop1 := evol.InitPopulation()
	pop2 := evol2.InitPopulation()

	for i := 0; i < 50; i++ {
		a := pop1[i%len(pop1)].Expr
		b := pop2[i%len(pop2)].Expr
		if child, ok := evol.Crossover(a, b); ok {
			if n := expression.Count(child); n > cfg.GenConfig.MaxNodes {
				t.Errorf("crossover %d: Count=%d > MaxNodes=%d", i, n, cfg.GenConfig.MaxNodes)
			}
		}
	}
}

func TestCrossover_ParentsUnchanged(t *testing.T) {
	evol, _ := genetic.NewEvolution(crossoverCfg(), testDS())
	a := expression.NewBinary(expression.NodeAdd, expression.NewVariable(), expression.NewConstant(3))
	b := expression.NewBinary(expression.NodeMul, expression.NewVariable(), expression.NewConstant(2))

	fmtA := expression.Format(a)
	fmtB := expression.Format(b)

	for i := 0; i < 10; i++ {
		evol.Crossover(a, b)
	}

	if expression.Format(a) != fmtA {
		t.Errorf("padre A modificado: %s → %s", fmtA, expression.Format(a))
	}
	if expression.Format(b) != fmtB {
		t.Errorf("padre B modificado: %s → %s", fmtB, expression.Format(b))
	}
}

func TestCrossover_NoAliasing(t *testing.T) {
	evol, _ := genetic.NewEvolution(crossoverCfg(), testDS())
	a := expression.NewBinary(expression.NodeAdd, expression.NewVariable(), expression.NewConstant(3))
	b := expression.NewBinary(expression.NodeMul, expression.NewVariable(), expression.NewConstant(2))

	var child *expression.Node
	for i := 0; i < 20 && child == nil; i++ {
		if c, ok := evol.Crossover(a, b); ok {
			child = c
		}
	}
	if child == nil {
		t.Skip("crossover rechazado en todos los intentos")
	}

	childBefore := expression.Format(child)

	// Modificar constantes de ambos padres
	walkAllTest(a, func(n *expression.Node) {
		if n.Type == expression.NodeConstant {
			n.Value = 99999
		}
	})
	walkAllTest(b, func(n *expression.Node) {
		if n.Type == expression.NodeConstant {
			n.Value = 88888
		}
	})

	// El hijo no debe haberse alterado
	if expression.Format(child) != childBefore {
		t.Errorf("el hijo comparte nodos con los padres (aliasing):\nantes: %s\ndespués: %s",
			childBefore, expression.Format(child))
	}
}

// ─── Mutation ─────────────────────────────────────────────────────────────

func TestMutate_ReturnsValidTree(t *testing.T) {
	evol, _ := genetic.NewEvolution(testCfg(), testDS())
	pop := evol.InitPopulation()
	for i, ind := range pop {
		m := evol.Mutate(ind.Expr)
		if m == nil {
			t.Errorf("Mutate %d: devolvió nil", i)
		}
	}
}

func TestMutate_RespectsMaxDepth(t *testing.T) {
	cfg := testCfg()
	evol, _ := genetic.NewEvolution(cfg, testDS())
	pop := evol.InitPopulation()
	for i := 0; i < 100; i++ {
		src := pop[i%len(pop)].Expr
		m := evol.Mutate(src)
		if d := expression.Depth(m); d > cfg.GenConfig.MaxDepth {
			t.Errorf("mutación %d: Depth=%d > MaxDepth=%d", i, d, cfg.GenConfig.MaxDepth)
		}
	}
}

func TestMutate_RespectsMaxNodes(t *testing.T) {
	cfg := testCfg()
	evol, _ := genetic.NewEvolution(cfg, testDS())
	pop := evol.InitPopulation()
	for i := 0; i < 100; i++ {
		src := pop[i%len(pop)].Expr
		m := evol.Mutate(src)
		if n := expression.Count(m); n > cfg.GenConfig.MaxNodes {
			t.Errorf("mutación %d: Count=%d > MaxNodes=%d", i, n, cfg.GenConfig.MaxNodes)
		}
	}
}

func TestMutate_OriginalUnchanged(t *testing.T) {
	evol, _ := genetic.NewEvolution(testCfg(), testDS())
	original := expression.NewBinary(expression.NodeAdd,
		expression.NewVariable(), expression.NewConstant(7))
	fmtBefore := expression.Format(original)

	for i := 0; i < 20; i++ {
		evol.Mutate(original)
	}
	if expression.Format(original) != fmtBefore {
		t.Errorf("Mutate modificó el árbol original: %s → %s",
			fmtBefore, expression.Format(original))
	}
}

// ─── Evolution ────────────────────────────────────────────────────────────

func TestEvolution_PopulationSize(t *testing.T) {
	cfg := testCfg()
	evol, _ := genetic.NewEvolution(cfg, testDS())
	pop := evol.InitPopulation()

	for gen := 0; gen < 5; gen++ {
		pop = evol.NextGeneration(pop)
		if len(pop) != cfg.PopulationSize {
			t.Errorf("gen %d: len(pop) = %d, want %d", gen, len(pop), cfg.PopulationSize)
		}
	}
}

func TestEvolution_GlobalBestNeverWorsens(t *testing.T) {
	cfg := testCfg()
	cfg.Generations = 20

	var history []float64
	cfg.OnGeneration = func(gen int, best genetic.Individual) {
		history = append(history, best.Result.Fitness)
	}

	evol, _ := genetic.NewEvolution(cfg, testDS())
	_, err := evol.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	for i := 1; i < len(history); i++ {
		if history[i] > history[i-1]+1e-10 {
			t.Errorf("gen %d: global best empeoró de %v a %v", i, history[i-1], history[i])
		}
	}
}

func TestEvolution_NoPanic(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("panic durante evolución: %v", r)
		}
	}()
	cfg := testCfg()
	evol, _ := genetic.NewEvolution(cfg, testDS())
	_, err := evol.Run()
	if err != nil {
		t.Fatalf("Run devolvió error: %v", err)
	}
}

func TestEvolution_Reproducible(t *testing.T) {
	cfg := testCfg()
	cfg.Generations = 10

	evol1, _ := genetic.NewEvolution(cfg, testDS())
	best1, _ := evol1.Run()

	evol2, _ := genetic.NewEvolution(cfg, testDS())
	best2, _ := evol2.Run()

	if expression.Format(best1.Expr) != expression.Format(best2.Expr) {
		t.Errorf("misma seed produce resultados distintos:\n  %s\n  %s",
			expression.Format(best1.Expr), expression.Format(best2.Expr))
	}
	if best1.Result.Fitness != best2.Result.Fitness {
		t.Errorf("misma seed produce fitness distintos: %v vs %v",
			best1.Result.Fitness, best2.Result.Fitness)
	}
}

// ─── Caso simple: y = 2*x + 1 ─────────────────────────────────────────────

func TestEvolution_LinearDataset(t *testing.T) {
	cfg := genetic.Config{
		PopulationSize: 100,
		Generations:    50,
		TournamentSize: 3,
		ElitismRate:    0.05,
		MutationRate:   0.10,
		CrossoverRate:  0.80,
		Lambda:         0.01,
		Seed:           1,
		GenConfig: generator.Config{
			MaxDepth:            5,
			MaxNodes:            30,
			AllowedBinaryOps:    []expression.NodeType{expression.NodeAdd, expression.NodeSub, expression.NodeMul, expression.NodeDiv},
			AllowedUnaryOps:     nil,
			ConstantMin:         -5,
			ConstantMax:         5,
			VariableProbability: 0.5,
			TerminalProbability: 0.3,
			Seed:                1,
		},
	}

	var initialMSE, finalMSE float64
	first := true
	cfg.OnGeneration = func(gen int, best genetic.Individual) {
		if first {
			initialMSE = best.Result.MSE
			first = false
		}
		finalMSE = best.Result.MSE
	}

	evol, err := genetic.NewEvolution(cfg, testDS())
	if err != nil {
		t.Fatalf("NewEvolution: %v", err)
	}
	best, err := evol.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// El GA debe producir algo razonablemente bueno (MSE < 100 es muy conservador)
	if best.Result.MSE > 100 {
		t.Errorf("MSE final %v es demasiado alto para y=2x+1 (esperado < 100)", best.Result.MSE)
	}
	// El MSE final no puede ser peor que el inicial (monotonía del global best)
	if finalMSE > initialMSE+1e-9 {
		t.Errorf("MSE empeoró: inicial=%v final=%v", initialMSE, finalMSE)
	}
}
