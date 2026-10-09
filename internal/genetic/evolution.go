package genetic

import (
	"fmt"
	"math/rand"

	"github.com/LautiSeverino/symbolic-regression/internal/dataset"
	"github.com/LautiSeverino/symbolic-regression/internal/expression"
	"github.com/LautiSeverino/symbolic-regression/internal/fitness"
	"github.com/LautiSeverino/symbolic-regression/internal/generator"
)

// Evolution ejecuta el algoritmo genético con evaluación de fitness paralela.
//
// La paralelización se limita al cálculo de fitness (evaluatePopulation).
// La generación de descendencia — selección, crossover, mutación — permanece
// secuencial y usa e.rng de forma exclusiva, lo que garantiza que misma seed +
// misma config produce exactamente el mismo resultado con cualquier Workers.
type Evolution struct {
	cfg Config
	rng *rand.Rand           // RNG del GA, exclusivo del goroutine principal
	ds  *dataset.Dataset     // dataset de training, read-only después de la construcción
	gen *generator.Generator // para población inicial y mutación de subárboles
}

// NewEvolution valida cfg, normaliza Workers, construye el generador y retorna
// una Evolution lista para usar.
func NewEvolution(cfg Config, ds *dataset.Dataset) (*Evolution, error) {
	// Normalizar Workers antes de validate: ≤ 0 → 1.
	if cfg.Workers <= 0 {
		cfg.Workers = 1
	}

	if err := cfg.validate(); err != nil {
		return nil, err
	}
	if ds == nil || ds.Len() == 0 {
		return nil, fmt.Errorf("genetic: dataset is nil or empty")
	}
	gen, err := generator.New(cfg.GenConfig)
	if err != nil {
		return nil, fmt.Errorf("genetic: invalid generator config: %w", err)
	}
	return &Evolution{
		cfg: cfg,
		rng: rand.New(rand.NewSource(cfg.Seed)),
		ds:  ds,
		gen: gen,
	}, nil
}

// InitPopulation genera la población inicial y evalúa el fitness de cada individuo.
//
// Separación generación / evaluación:
//  1. Las expresiones se generan con e.gen de forma secuencial (determinista).
//  2. El fitness de todas las expresiones se calcula en paralelo con evaluatePopulation.
//  3. Los resultados se ensamblan en el orden original.
//
// Esta separación preserva la secuencia de e.gen independientemente de Workers,
// por lo que la población generada es siempre idéntica para la misma seed.
func (e *Evolution) InitPopulation() []Individual {
	// Fase 1 — generación secuencial (usa e.gen, determinista).
	exprs := make([]*expression.Node, e.cfg.PopulationSize)
	for i := range exprs {
		exprs[i] = e.gen.Generate()
	}

	// Fase 2 — evaluación paralela (pura, sin estado mutable compartido).
	results := e.evaluatePopulation(exprs)

	// Fase 3 — ensamblado.
	pop := make([]Individual, e.cfg.PopulationSize)
	for i := range pop {
		pop[i] = Individual{Expr: exprs[i], Result: results[i]}
	}
	return pop
}

// Run ejecuta el loop evolutivo y retorna el mejor individuo global.
//
// El globalBest nunca empeora: si gen N no produce nada mejor, se conserva
// el mejor encontrado hasta ese momento (garantizado por elitismo + tracking).
//
// OnGeneration (si no es nil) se invoca:
//   - gen=0: con la población inicial
//   - gen=1..Generations: con la generación evolucionada
func (e *Evolution) Run() (*Individual, error) {
	pop := e.InitPopulation()
	sortByFitness(pop)

	globalBest := cloneIndividual(pop[0])
	if e.cfg.OnGeneration != nil {
		e.cfg.OnGeneration(0, globalBest)
	}

	for gen := 1; gen <= e.cfg.Generations; gen++ {
		pop = e.NextGeneration(pop)
		sortByFitness(pop)

		if pop[0].Result.Fitness < globalBest.Result.Fitness {
			globalBest = cloneIndividual(pop[0])
		}
		if e.cfg.OnGeneration != nil {
			e.cfg.OnGeneration(gen, globalBest)
		}
	}

	return &globalBest, nil
}

// NextGeneration construye la siguiente generación aplicando elitismo, crossover
// y mutación. Produce exactamente PopulationSize individuos.
//
// Separación generación / evaluación:
//  1. Los élites se copian con sus resultados existentes (sin re-evaluar).
//  2. Las expresiones hijo se generan con e.rng de forma secuencial (determinista).
//  3. Solo las expresiones hijo nuevas se evalúan en paralelo con evaluatePopulation.
//  4. Los resultados se ensamblan en el orden original.
//
// Esta separación garantiza que la secuencia de e.rng es idéntica con cualquier
// Workers, por lo que los hijos generados son siempre los mismos para la misma seed.
//
// Un crossover rechazado (hijo viola límites) resulta en una copia del padre A.
func (e *Evolution) NextGeneration(pop []Individual) []Individual {
	sortByFitness(pop)

	// Fase 1a — élites: se copian con sus resultados existentes.
	elites := e.cfg.eliteCount()
	next := make([]Individual, 0, e.cfg.PopulationSize)
	for i := 0; i < elites && i < len(pop); i++ {
		next = append(next, cloneIndividual(pop[i]))
	}

	// Fase 1b — generación de descendencia: secuencial, usa e.rng.
	// La secuencia de llamadas a e.rng es idéntica independientemente de Workers.
	childCount := e.cfg.PopulationSize - len(next)
	childExprs := make([]*expression.Node, 0, childCount)
	for len(childExprs) < childCount {
		pA := e.TournamentSelect(pop)

		var childExpr *expression.Node
		if e.rng.Float64() < e.cfg.CrossoverRate {
			pB := e.TournamentSelect(pop)
			if child, ok := e.Crossover(pA.Expr, pB.Expr); ok {
				childExpr = child
			} else {
				childExpr = expression.Clone(pA.Expr)
			}
		} else {
			childExpr = expression.Clone(pA.Expr)
		}

		if e.rng.Float64() < e.cfg.MutationRate {
			childExpr = e.Mutate(childExpr)
		}

		childExprs = append(childExprs, childExpr)
	}

	// Fase 2 — evaluación paralela de los hijos nuevos.
	childResults := e.evaluatePopulation(childExprs)

	// Fase 3 — ensamblado de la nueva generación.
	for i, expr := range childExprs {
		next = append(next, Individual{Expr: expr, Result: childResults[i]})
	}

	return next
}

// evalExpr evalúa expr sobre el dataset de training.
// Solo lee e.ds y e.cfg.Lambda — safe para llamadas concurrentes desde workers.
// Ante errores de fitness retorna la penalización máxima.
func (e *Evolution) evalExpr(expr *expression.Node) fitness.FitnessResult {
	result, err := fitness.Evaluate(expr, e.ds, e.cfg.Lambda)
	if err != nil {
		return fitness.FitnessResult{
			MSE:     fitness.PenaltyPerInvalidPoint,
			Fitness: fitness.PenaltyPerInvalidPoint,
		}
	}
	return result
}
