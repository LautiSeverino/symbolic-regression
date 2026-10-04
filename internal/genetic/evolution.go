package genetic

import (
	"fmt"
	"math/rand"

	"github.com/LautiSeverino/symbolic-regression/internal/dataset"
	"github.com/LautiSeverino/symbolic-regression/internal/expression"
	"github.com/LautiSeverino/symbolic-regression/internal/fitness"
	"github.com/LautiSeverino/symbolic-regression/internal/generator"
)

// Evolution ejecuta el algoritmo genético de forma completamente secuencial.
// No utiliza goroutines, channels ni worker pool.
type Evolution struct {
	cfg Config
	rng *rand.Rand           // RNG del GA, independiente del generador
	ds  *dataset.Dataset     // dataset de training
	gen *generator.Generator // para población inicial y mutación de subárboles
}

// NewEvolution valida cfg, construye el generador y retorna una Evolution lista.
func NewEvolution(cfg Config, ds *dataset.Dataset) (*Evolution, error) {
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
func (e *Evolution) InitPopulation() []Individual {
	pop := make([]Individual, e.cfg.PopulationSize)
	for i := range pop {
		expr := e.gen.Generate()
		pop[i] = Individual{Expr: expr, Result: e.evalExpr(expr)}
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

// NextGeneration construye la siguiente generación aplicando elitismo, crossover y mutación.
// Produce exactamente PopulationSize individuos.
// Un crossover rechazado (hijo viola límites) resulta en una copia del padre A.
// Esta implementación produce un hijo por operación de crossover.
func (e *Evolution) NextGeneration(pop []Individual) []Individual {
	sortByFitness(pop)
	next := make([]Individual, 0, e.cfg.PopulationSize)

	// Elitismo: los mejores individuos sobreviven intactos con expresión clonada
	elites := e.cfg.eliteCount()
	for i := 0; i < elites && i < len(pop); i++ {
		next = append(next, cloneIndividual(pop[i]))
	}

	// Rellenar la nueva generación con selección → crossover → mutación
	for len(next) < e.cfg.PopulationSize {
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

		next = append(next, Individual{Expr: childExpr, Result: e.evalExpr(childExpr)})
	}

	return next
}

// evalExpr evalúa expr sobre el dataset de training.
// Ante errores de fitness (lambda no finito, etc.) retorna la penalización máxima.
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
