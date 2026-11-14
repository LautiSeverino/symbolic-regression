package genetic_test

import (
	"math"
	"sort"
	"testing"

	"github.com/LautiSeverino/symbolic-regression/internal/dataset"
	"github.com/LautiSeverino/symbolic-regression/internal/expression"
	"github.com/LautiSeverino/symbolic-regression/internal/fitness"
	"github.com/LautiSeverino/symbolic-regression/internal/generator"
	"github.com/LautiSeverino/symbolic-regression/internal/genetic"
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
		Workers:        1,
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
	evol, err := genetic.NewEvolution(testCfg(), testDS(), nil)
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
			c := base
			c.TournamentSize = base.PopulationSize + 1
			return c
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
			_, err := genetic.NewEvolution(tc.cfg, ds, nil)
			if err == nil {
				t.Error("se esperaba error para config/dataset inválido, se obtuvo nil")
			}
		})
	}
}

// ─── Validación del dataset de validación ─────────────────────────────────

func TestNewEvolution_ValidationDatasetValidation(t *testing.T) {
	train := testDS()

	t.Run("nil val es válido", func(t *testing.T) {
		_, err := genetic.NewEvolution(testCfg(), train, nil)
		if err != nil {
			t.Errorf("val=nil: error inesperado: %v", err)
		}
	})

	t.Run("val con puntos es válido", func(t *testing.T) {
		val := dataset.New([]dataset.Point{{X: 6, Y: 13}, {X: 7, Y: 15}})
		_, err := genetic.NewEvolution(testCfg(), train, val)
		if err != nil {
			t.Errorf("val con puntos: error inesperado: %v", err)
		}
	})

	t.Run("val vacío es error", func(t *testing.T) {
		val := dataset.New(nil) // non-nil, 0 puntos
		_, err := genetic.NewEvolution(testCfg(), train, val)
		if err == nil {
			t.Error("val vacío: se esperaba error, se obtuvo nil")
		}
	})
}

// ─── Population ───────────────────────────────────────────────────────────

func TestInitPopulation_Size(t *testing.T) {
	cfg := testCfg()
	evol, _ := genetic.NewEvolution(cfg, testDS(), nil)
	pop := evol.InitPopulation()
	if len(pop) != cfg.PopulationSize {
		t.Errorf("len(pop) = %d, want %d", len(pop), cfg.PopulationSize)
	}
}

func TestInitPopulation_ValidExpr(t *testing.T) {
	evol, _ := genetic.NewEvolution(testCfg(), testDS(), nil)
	for i, ind := range evol.InitPopulation() {
		if ind.Expr == nil {
			t.Errorf("individuo %d: Expr es nil", i)
		}
	}
}

func TestInitPopulation_FitnessComputed(t *testing.T) {
	evol, _ := genetic.NewEvolution(testCfg(), testDS(), nil)
	for i, ind := range evol.InitPopulation() {
		if math.IsNaN(ind.Result.Fitness) || math.IsInf(ind.Result.Fitness, 0) {
			t.Errorf("individuo %d: Fitness no finito: %v", i, ind.Result.Fitness)
		}
	}
}

// ─── Tournament Selection ─────────────────────────────────────────────────

func TestTournamentSelect_FromPop(t *testing.T) {
	evol, _ := genetic.NewEvolution(testCfg(), testDS(), nil)
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
	evol, _ := genetic.NewEvolution(testCfg(), testDS(), nil)

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
	evol, _ := genetic.NewEvolution(cfg, testDS(), nil)

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
	evol, _ := genetic.NewEvolution(cfg, testDS(), nil)

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
	evol, _ := genetic.NewEvolution(crossoverCfg(), testDS(), nil)
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
	evol, _ := genetic.NewEvolution(cfg, testDS(), nil)
	evol2, _ := genetic.NewEvolution(crossoverCfg(), testDS(), nil)

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
	evol, _ := genetic.NewEvolution(cfg, testDS(), nil)
	evol2, _ := genetic.NewEvolution(crossoverCfg(), testDS(), nil)

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
	evol, _ := genetic.NewEvolution(crossoverCfg(), testDS(), nil)
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
	evol, _ := genetic.NewEvolution(crossoverCfg(), testDS(), nil)
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
	evol, _ := genetic.NewEvolution(testCfg(), testDS(), nil)
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
	evol, _ := genetic.NewEvolution(cfg, testDS(), nil)
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
	evol, _ := genetic.NewEvolution(cfg, testDS(), nil)
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
	evol, _ := genetic.NewEvolution(testCfg(), testDS(), nil)
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
	evol, _ := genetic.NewEvolution(cfg, testDS(), nil)
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

	evol, _ := genetic.NewEvolution(cfg, testDS(), nil)
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
	evol, _ := genetic.NewEvolution(cfg, testDS(), nil)
	_, err := evol.Run()
	if err != nil {
		t.Fatalf("Run devolvió error: %v", err)
	}
}

func TestEvolution_Reproducible(t *testing.T) {
	cfg := testCfg()
	cfg.Generations = 10

	evol1, _ := genetic.NewEvolution(cfg, testDS(), nil)
	best1, _ := evol1.Run()

	evol2, _ := genetic.NewEvolution(cfg, testDS(), nil)
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

	evol, err := genetic.NewEvolution(cfg, testDS(), nil)
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

// ─── ValidationResult en Run() ────────────────────────────────────────────

func TestRun_ValidationResultNilWhenNoVal(t *testing.T) {
	evol, _ := genetic.NewEvolution(testCfg(), testDS(), nil)
	best, err := evol.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if best.ValidationResult != nil {
		t.Error("ValidationResult debe ser nil cuando val es nil")
	}
}

func TestRun_ValidationResultSetWhenVal(t *testing.T) {
	val := dataset.New([]dataset.Point{{X: 6, Y: 13}, {X: 7, Y: 15}})
	evol, _ := genetic.NewEvolution(testCfg(), testDS(), val)
	best, err := evol.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if best.ValidationResult == nil {
		t.Error("ValidationResult debe ser no-nil cuando val es no-nil")
	}
}

func TestRun_ValidationResultFinite(t *testing.T) {
	val := dataset.New([]dataset.Point{{X: 6, Y: 13}, {X: 7, Y: 15}})
	evol, _ := genetic.NewEvolution(testCfg(), testDS(), val)
	best, err := evol.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if best.ValidationResult == nil {
		t.Fatal("ValidationResult es nil")
	}
	if math.IsNaN(best.ValidationResult.MSE) || math.IsInf(best.ValidationResult.MSE, 0) {
		t.Errorf("ValidationResult.MSE no finito: %v", best.ValidationResult.MSE)
	}
}

func TestRun_ValidationLambdaZero(t *testing.T) {
	// Con lambda=0: Fitness = MSE + 0*Complexity = MSE (sin penalización).
	// Esto verifica que Run() evalúa la validación con lambda=0.
	val := dataset.New([]dataset.Point{{X: 6, Y: 13}, {X: 7, Y: 15}})
	evol, _ := genetic.NewEvolution(testCfg(), testDS(), val)
	best, err := evol.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if best.ValidationResult == nil {
		t.Fatal("ValidationResult es nil")
	}
	if best.ValidationResult.Fitness != best.ValidationResult.MSE {
		t.Errorf("con lambda=0: Fitness(%v) debe ser igual a MSE(%v)",
			best.ValidationResult.Fitness, best.ValidationResult.MSE)
	}
}

func TestRun_ValidationDoesNotAffectTraining(t *testing.T) {
	// Test crítico: misma configuración, semilla y training deben producir
	// el mismo resultado con y sin validación.
	// La validación no debe alterar el RNG, la selección ni el fitness evolutivo.
	cfg := testCfg()
	train := testDS()
	val := dataset.New([]dataset.Point{{X: 6, Y: 13}, {X: 7, Y: 15}})

	// Con validación
	evol1, err := genetic.NewEvolution(cfg, train, val)
	if err != nil {
		t.Fatalf("NewEvolution con val: %v", err)
	}
	best1, err := evol1.Run()
	if err != nil {
		t.Fatalf("Run con val: %v", err)
	}

	// Sin validación (misma seed)
	evol2, err := genetic.NewEvolution(cfg, train, nil)
	if err != nil {
		t.Fatalf("NewEvolution sin val: %v", err)
	}
	best2, err := evol2.Run()
	if err != nil {
		t.Fatalf("Run sin val: %v", err)
	}

	// Los resultados de entrenamiento deben ser idénticos
	if expression.Format(best1.Expr) != expression.Format(best2.Expr) {
		t.Errorf("el validation cambió el resultado de training:\n  con val: %s\n  sin val: %s",
			expression.Format(best1.Expr), expression.Format(best2.Expr))
	}
	if best1.Result.Fitness != best2.Result.Fitness {
		t.Errorf("el validation cambió el fitness de training: %v vs %v",
			best1.Result.Fitness, best2.Result.Fitness)
	}
	if best1.Result.MSE != best2.Result.MSE {
		t.Errorf("el validation cambió el MSE de training: %v vs %v",
			best1.Result.MSE, best2.Result.MSE)
	}

	// Solo el que tiene val debe tener ValidationResult no-nil
	if best1.ValidationResult == nil {
		t.Error("con val: ValidationResult debe ser no-nil")
	}
	if best2.ValidationResult != nil {
		t.Error("sin val: ValidationResult debe ser nil")
	}
}

func TestRun_ValidationReproducible(t *testing.T) {
	// Misma seed → mismo ValidationResult.MSE en dos ejecuciones.
	cfg := testCfg()
	train := testDS()
	val := dataset.New([]dataset.Point{{X: 6, Y: 13}, {X: 7, Y: 15}})

	run := func() float64 {
		evol, err := genetic.NewEvolution(cfg, train, val)
		if err != nil {
			t.Fatalf("NewEvolution: %v", err)
		}
		best, err := evol.Run()
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		if best.ValidationResult == nil {
			t.Fatal("ValidationResult nil")
		}
		return best.ValidationResult.MSE
	}

	mse1 := run()
	mse2 := run()

	if mse1 != mse2 {
		t.Errorf("validation MSE no reproducible: %v vs %v", mse1, mse2)
	}
}

func TestValidationEvaluate_PartialInvalid(t *testing.T) {
	// Verifica que el mecanismo de evaluación de validación (fitness.Evaluate
	// con lambda=0) maneja correctamente expresiones que fallan en algunos puntos.
	//
	// Usa ln(x) como expresión controlada: falla determinísticamente para x ≤ 0.
	// No depende del algoritmo genético para generar la expresión.
	lnX := expression.NewUnary(expression.NodeLn, expression.NewVariable())

	// 2 puntos inválidos (x ≤ 0) y 2 válidos (x > 0)
	val := dataset.New([]dataset.Point{
		{X: -1, Y: 0}, // inválido: ln(-1)
		{X: 0, Y: 0},  // inválido: ln(0)
		{X: 1, Y: 0},  // válido: ln(1) = 0
		{X: 2, Y: 0},  // válido: ln(2) ≈ 0.693
	})

	// fitness.Evaluate con lambda=0 es exactamente lo que Run() usa para validación
	result, err := fitness.Evaluate(lnX, val, 0)
	if err != nil {
		t.Fatalf("Evaluate inesperado: %v", err)
	}
	if result.InvalidCount != 2 {
		t.Errorf("InvalidCount = %d, want 2", result.InvalidCount)
	}
	if math.IsNaN(result.MSE) || math.IsInf(result.MSE, 0) {
		t.Errorf("MSE debe ser finito con puntos inválidos: %v", result.MSE)
	}
	if result.MSE <= 0 {
		t.Errorf("MSE debe ser > 0 por la penalización: %v", result.MSE)
	}
	// Con lambda=0: Fitness == MSE exacto (sin penalización de complejidad)
	if result.Fitness != result.MSE {
		t.Errorf("con lambda=0: Fitness(%v) debe ser igual a MSE(%v)",
			result.Fitness, result.MSE)
	}
}

// ─── Helpers Phase 14 ────────────────────────────────────────────────────

// stagnationTestCfg devuelve una Config con MutationRate=0 y CrossoverRate=0.
// Con estas tasas, todos los hijos son copias exactas de individuos seleccionados;
// ninguna generación puede producir un individuo mejor que el mejor actual.
// Por lo tanto stagnationCount incrementa cada generación de forma determinista.
func stagnationTestCfg(limit int) genetic.Config {
	return genetic.Config{
		PopulationSize:  20,
		Generations:     100,
		TournamentSize:  2,
		ElitismRate:     0.05,
		MutationRate:    0, // sin mutación → hijos son copias exactas
		CrossoverRate:   0, // sin crossover → hijos son copias exactas
		Lambda:          0.01,
		Seed:            42,
		Workers:         1,
		StagnationLimit: limit,
		StagnationDelta: 0,
		GenConfig: generator.Config{
			MaxDepth:            3,
			MaxNodes:            10,
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

// ─── Validación de Config — campos nuevos ────────────────────────────────

func TestConfig_Validate_TargetMSE_NaN_AlwaysError(t *testing.T) {
	// NaN se rechaza siempre, sin importar TargetMSEEnabled.
	for _, enabled := range []bool{false, true} {
		cfg := testCfg()
		cfg.TargetMSEEnabled = enabled
		cfg.TargetMSE = math.NaN()
		_, err := genetic.NewEvolution(cfg, testDS(), nil)
		if err == nil {
			t.Errorf("TargetMSEEnabled=%v, TargetMSE=NaN: se esperaba error, se obtuvo nil", enabled)
		}
	}
}

func TestConfig_Validate_TargetMSE_Enabled_NegativeError(t *testing.T) {
	cfg := testCfg()
	cfg.TargetMSEEnabled = true
	cfg.TargetMSE = -1
	_, err := genetic.NewEvolution(cfg, testDS(), nil)
	if err == nil {
		t.Error("TargetMSEEnabled=true, TargetMSE=-1: se esperaba error")
	}
}

func TestConfig_Validate_TargetMSE_Enabled_PlusInfError(t *testing.T) {
	cfg := testCfg()
	cfg.TargetMSEEnabled = true
	cfg.TargetMSE = math.Inf(1)
	_, err := genetic.NewEvolution(cfg, testDS(), nil)
	if err == nil {
		t.Error("TargetMSEEnabled=true, TargetMSE=+Inf: se esperaba error")
	}
}

func TestConfig_Validate_TargetMSE_Enabled_ZeroValid(t *testing.T) {
	// MSE=0 es un objetivo legítimo (alcanzable en datasets sintéticos exactos).
	cfg := testCfg()
	cfg.TargetMSEEnabled = true
	cfg.TargetMSE = 0
	_, err := genetic.NewEvolution(cfg, testDS(), nil)
	if err != nil {
		t.Errorf("TargetMSEEnabled=true, TargetMSE=0: error inesperado: %v", err)
	}
}

func TestConfig_Validate_TargetMSE_Disabled_ValueIgnored(t *testing.T) {
	// Con TargetMSEEnabled=false, cualquier valor (excepto NaN) es aceptado.
	cfg := testCfg()
	cfg.TargetMSEEnabled = false
	cfg.TargetMSE = -99
	_, err := genetic.NewEvolution(cfg, testDS(), nil)
	if err != nil {
		t.Errorf("TargetMSEEnabled=false, TargetMSE=-99: error inesperado: %v", err)
	}
}

func TestConfig_Validate_TargetMSE_Disabled_NegInfIgnored(t *testing.T) {
	// -Inf con TargetMSEEnabled=false: el valor es ignorado → válido.
	cfg := testCfg()
	cfg.TargetMSEEnabled = false
	cfg.TargetMSE = math.Inf(-1)
	_, err := genetic.NewEvolution(cfg, testDS(), nil)
	if err != nil {
		t.Errorf("TargetMSEEnabled=false, TargetMSE=-Inf: error inesperado: %v", err)
	}
}

func TestConfig_Validate_StagnationLimit_NegativeError(t *testing.T) {
	cfg := testCfg()
	cfg.StagnationLimit = -1
	_, err := genetic.NewEvolution(cfg, testDS(), nil)
	if err == nil {
		t.Error("StagnationLimit=-1: se esperaba error")
	}
}

func TestConfig_Validate_StagnationLimit_ZeroValid(t *testing.T) {
	cfg := testCfg()
	cfg.StagnationLimit = 0
	_, err := genetic.NewEvolution(cfg, testDS(), nil)
	if err != nil {
		t.Errorf("StagnationLimit=0: error inesperado: %v", err)
	}
}

func TestConfig_Validate_StagnationDelta_NegativeError(t *testing.T) {
	cfg := testCfg()
	cfg.StagnationDelta = -0.1
	_, err := genetic.NewEvolution(cfg, testDS(), nil)
	if err == nil {
		t.Error("StagnationDelta=-0.1: se esperaba error")
	}
}

func TestConfig_Validate_StagnationDelta_NaNError(t *testing.T) {
	cfg := testCfg()
	cfg.StagnationDelta = math.NaN()
	_, err := genetic.NewEvolution(cfg, testDS(), nil)
	if err == nil {
		t.Error("StagnationDelta=NaN: se esperaba error")
	}
}

func TestConfig_Validate_StagnationDelta_ZeroValid(t *testing.T) {
	cfg := testCfg()
	cfg.StagnationDelta = 0
	_, err := genetic.NewEvolution(cfg, testDS(), nil)
	if err != nil {
		t.Errorf("StagnationDelta=0: error inesperado: %v", err)
	}
}

// ─── Compatibilidad hacia atrás — criterios deshabilitados ───────────────

func TestEarlyStop_AllDisabled_RunsExactlyGenerations(t *testing.T) {
	// Con TargetMSEEnabled=false y StagnationLimit=0 (zeros values del struct),
	// el comportamiento debe ser idéntico al de Fase 13: corre exactamente
	// cfg.Generations generaciones. Usa testCfg() que omite los campos nuevos.
	cfg := testCfg() // TargetMSEEnabled=false, StagnationLimit=0 por zero value
	cfg.Generations = 5

	evol, err := genetic.NewEvolution(cfg, testDS(), nil)
	if err != nil {
		t.Fatalf("NewEvolution: %v", err)
	}
	best, err := evol.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if best.GenerationsRun != 5 {
		t.Errorf("GenerationsRun = %d, want 5 (criterios deshabilitados → runs completos)", best.GenerationsRun)
	}
}

// ─── GenerationsRun ───────────────────────────────────────────────────────

func TestGenerationsRun_ZeroDefaultValue(t *testing.T) {
	// El zero value de Individual debe tener GenerationsRun = 0.
	var ind genetic.Individual
	if ind.GenerationsRun != 0 {
		t.Errorf("GenerationsRun zero value = %d, want 0", ind.GenerationsRun)
	}
}

func TestGenerationsRun_EqualToGenerations_WhenNoEarlyStop(t *testing.T) {
	cfg := testCfg()
	cfg.Generations = 7

	evol, _ := genetic.NewEvolution(cfg, testDS(), nil)
	best, err := evol.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if best.GenerationsRun != 7 {
		t.Errorf("GenerationsRun = %d, want 7", best.GenerationsRun)
	}
}

// ─── TargetMSE en gen=0 ───────────────────────────────────────────────────

func TestEarlyStop_TargetMSE_Gen0_Fires(t *testing.T) {
	// math.MaxFloat64 es el mayor float64 finito. Cualquier MSE producido por
	// fitness.Evaluate es <= math.MaxFloat64, así que el criterio siempre se
	// cumple en gen=0 y el loop evolutivo nunca se ejecuta.
	cfg := testCfg()
	cfg.TargetMSEEnabled = true
	cfg.TargetMSE = math.MaxFloat64
	cfg.Generations = 50

	evol, err := genetic.NewEvolution(cfg, testDS(), nil)
	if err != nil {
		t.Fatalf("NewEvolution: %v", err)
	}
	best, err := evol.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if best.GenerationsRun != 0 {
		t.Errorf("GenerationsRun = %d, want 0 (parada en gen=0)", best.GenerationsRun)
	}
}

func TestEarlyStop_TargetMSE_Gen0_OnGenerationCalledOnce(t *testing.T) {
	// La parada en gen=0 ocurre DESPUÉS de OnGeneration(0,...).
	// Por lo tanto el callback debe haberse invocado exactamente una vez (gen=0).
	cfg := testCfg()
	cfg.TargetMSEEnabled = true
	cfg.TargetMSE = math.MaxFloat64
	cfg.Generations = 50

	var gensCalled []int
	cfg.OnGeneration = func(gen int, best genetic.Individual) {
		gensCalled = append(gensCalled, gen)
	}

	evol, _ := genetic.NewEvolution(cfg, testDS(), nil)
	_, err := evol.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(gensCalled) != 1 {
		t.Errorf("OnGeneration invocado %d veces, want 1 (solo gen=0)", len(gensCalled))
	}
	if len(gensCalled) == 1 && gensCalled[0] != 0 {
		t.Errorf("OnGeneration invocado con gen=%d, want 0", gensCalled[0])
	}
}

func TestEarlyStop_TargetMSE_Gen0_ValidationEvaluated(t *testing.T) {
	// Validation debe evaluarse aunque el loop no haya corrido (GenerationsRun=0).
	cfg := testCfg()
	cfg.TargetMSEEnabled = true
	cfg.TargetMSE = math.MaxFloat64

	val := dataset.New([]dataset.Point{{X: 6, Y: 13}, {X: 7, Y: 15}})
	evol, _ := genetic.NewEvolution(cfg, testDS(), val)
	best, err := evol.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if best.GenerationsRun != 0 {
		t.Errorf("GenerationsRun = %d, want 0", best.GenerationsRun)
	}
	if best.ValidationResult == nil {
		t.Fatal("ValidationResult debe ser no-nil aunque el loop no haya corrido")
	}
	if math.IsNaN(best.ValidationResult.MSE) || math.IsInf(best.ValidationResult.MSE, 0) {
		t.Errorf("ValidationResult.MSE no finito: %v", best.ValidationResult.MSE)
	}
}

// ─── Estancamiento — determinista con MutationRate=0, CrossoverRate=0 ───

func TestEarlyStop_Stagnation_Fires_AtExactLimit(t *testing.T) {
	// Con MutationRate=0 y CrossoverRate=0 los hijos son copias exactas de
	// individuos existentes; el globalBest nunca mejora. Por tanto
	// stagnationCount incrementa cada generación y el loop termina exactamente
	// en gen=StagnationLimit.
	cfg := stagnationTestCfg(5)

	evol, err := genetic.NewEvolution(cfg, testDS(), nil)
	if err != nil {
		t.Fatalf("NewEvolution: %v", err)
	}
	best, err := evol.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if best.GenerationsRun != 5 {
		t.Errorf("GenerationsRun = %d, want 5 (StagnationLimit=5, MutationRate=CrossoverRate=0)",
			best.GenerationsRun)
	}
}

func TestEarlyStop_Stagnation_Zero_DoesNotFire(t *testing.T) {
	// StagnationLimit=0 deshabilita el criterio → corre hasta cfg.Generations.
	cfg := stagnationTestCfg(0)
	cfg.Generations = 5

	evol, _ := genetic.NewEvolution(cfg, testDS(), nil)
	best, err := evol.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if best.GenerationsRun != 5 {
		t.Errorf("GenerationsRun = %d, want 5 (StagnationLimit=0 deshabilitado)", best.GenerationsRun)
	}
}

func TestEarlyStop_Stagnation_LimitOne_FiresAtFirstNonImprovement(t *testing.T) {
	// StagnationLimit=1: para en la primera generación sin mejora.
	// Con MutationRate=0 y CrossoverRate=0, eso ocurre en gen=1.
	cfg := stagnationTestCfg(1)

	evol, _ := genetic.NewEvolution(cfg, testDS(), nil)
	best, err := evol.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if best.GenerationsRun != 1 {
		t.Errorf("GenerationsRun = %d, want 1 (StagnationLimit=1)", best.GenerationsRun)
	}
}

func TestEarlyStop_Stagnation_Deterministic(t *testing.T) {
	// Dos runs con la misma seed deben producir exactamente el mismo GenerationsRun.
	cfg := stagnationTestCfg(5)

	run := func() int {
		evol, err := genetic.NewEvolution(cfg, testDS(), nil)
		if err != nil {
			t.Fatalf("NewEvolution: %v", err)
		}
		best, err := evol.Run()
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		return best.GenerationsRun
	}

	r1 := run()
	r2 := run()
	if r1 != r2 {
		t.Errorf("GenerationsRun distinto entre runs: %d vs %d", r1, r2)
	}
}

func TestEarlyStop_Stagnation_WorkersInvariant(t *testing.T) {
	// El determinismo es independiente del número de Workers.
	run := func(workers int) int {
		cfg := stagnationTestCfg(5)
		cfg.Workers = workers
		evol, err := genetic.NewEvolution(cfg, testDS(), nil)
		if err != nil {
			t.Fatalf("NewEvolution workers=%d: %v", workers, err)
		}
		best, err := evol.Run()
		if err != nil {
			t.Fatalf("Run workers=%d: %v", workers, err)
		}
		return best.GenerationsRun
	}

	r1 := run(1)
	r4 := run(4)
	if r1 != r4 {
		t.Errorf("GenerationsRun: Workers=1→%d, Workers=4→%d (deben ser iguales)", r1, r4)
	}
}

// ─── Criterios simultáneos ────────────────────────────────────────────────

func TestEarlyStop_TargetMSE_Gen0_Fires_BeforeLoop(t *testing.T) {
	// Verifica que TargetMSE=math.MaxFloat64 para en gen=0 (antes del loop),
	// independientemente de StagnationLimit. Este test no es sobre prioridad
	// entre criterios dentro del loop (ambos se evalúan en el loop), sino sobre
	// que el check de gen=0 impide que el loop se ejecute en absoluto.
	cfg := stagnationTestCfg(5) // StagnationLimit=5
	cfg.TargetMSEEnabled = true
	cfg.TargetMSE = math.MaxFloat64
	cfg.Generations = 100

	evol, _ := genetic.NewEvolution(cfg, testDS(), nil)
	best, err := evol.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	// TargetMSE se cumple en gen=0 → loop no ejecutado → GenerationsRun=0.
	if best.GenerationsRun != 0 {
		t.Errorf("GenerationsRun = %d, want 0 (TargetMSE=MaxFloat64 para en gen=0)", best.GenerationsRun)
	}
}

func TestEarlyStop_OnlyStagnation_Fires(t *testing.T) {
	// TargetMSE deshabilitado; solo estancamiento activo. Con MutationRate=0
	// y CrossoverRate=0 el estancamiento dispara exactamente en gen=StagnationLimit.
	cfg := stagnationTestCfg(3)
	cfg.TargetMSEEnabled = false

	evol, _ := genetic.NewEvolution(cfg, testDS(), nil)
	best, err := evol.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if best.GenerationsRun != 3 {
		t.Errorf("GenerationsRun = %d, want 3 (StagnationLimit=3)", best.GenerationsRun)
	}
}

// ─── Validación post early stop ───────────────────────────────────────────

func TestEarlyStop_ValidationEvaluated_AfterGen0Stop(t *testing.T) {
	cfg := testCfg()
	cfg.TargetMSEEnabled = true
	cfg.TargetMSE = math.MaxFloat64

	val := dataset.New([]dataset.Point{{X: 6, Y: 13}, {X: 7, Y: 15}})
	evol, _ := genetic.NewEvolution(cfg, testDS(), val)
	best, err := evol.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if best.ValidationResult == nil {
		t.Error("ValidationResult debe ser no-nil después de parada en gen=0")
	}
}

func TestEarlyStop_ValidationEvaluated_AfterStagnationStop(t *testing.T) {
	cfg := stagnationTestCfg(3)
	val := dataset.New([]dataset.Point{{X: 6, Y: 13}, {X: 7, Y: 15}})

	evol, err := genetic.NewEvolution(cfg, testDS(), val)
	if err != nil {
		t.Fatalf("NewEvolution: %v", err)
	}
	best, err := evol.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if best.ValidationResult == nil {
		t.Error("ValidationResult debe ser no-nil después de parada por estancamiento")
	}
	if math.IsNaN(best.ValidationResult.MSE) || math.IsInf(best.ValidationResult.MSE, 0) {
		t.Errorf("ValidationResult.MSE no finito: %v", best.ValidationResult.MSE)
	}
}

func TestEarlyStop_OnGeneration_CalledForTerminatingGeneration(t *testing.T) {
	// La generación que gatilla el estancamiento debe ser reportada al callback
	// antes de salir del loop.
	cfg := stagnationTestCfg(3)

	var lastGenCalled int
	cfg.OnGeneration = func(gen int, best genetic.Individual) {
		lastGenCalled = gen
	}

	evol, _ := genetic.NewEvolution(cfg, testDS(), nil)
	best, err := evol.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	// La última generación llamada en el callback debe coincidir con GenerationsRun.
	if lastGenCalled != best.GenerationsRun {
		t.Errorf("última gen en callback = %d, GenerationsRun = %d (deben coincidir)",
			lastGenCalled, best.GenerationsRun)
	}
}
