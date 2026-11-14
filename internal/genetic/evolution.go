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
// ds es exclusivamente el dataset de entrenamiento.
// val, si no es nil, se usa para una única evaluación post-evolución sobre
// el mejor individuo final. Nunca participa en selección, crossover,
// mutación, cálculo de fitness evolutivo ni en el callback OnGeneration.
type Evolution struct {
	cfg Config
	rng *rand.Rand           // RNG del GA, exclusivo del goroutine principal
	ds  *dataset.Dataset     // dataset de entrenamiento (read-only post-construcción)
	val *dataset.Dataset     // dataset de validación; nil = sin validación
	gen *generator.Generator // para población inicial y mutación de subárboles
}

// NewEvolution valida cfg, normaliza Workers, construye el generador y retorna
// una Evolution lista para usar.
//
// train es el dataset de entrenamiento (obligatorio, no nil ni vacío).
// val es el dataset de validación (opcional):
//   - nil   → sin validación; Individual.ValidationResult siempre nil.
//   - vacío → error explícito (use nil para omitir validación).
//
// val nunca afecta la evolución: selección, crossover, mutación y fitness
// operan exclusivamente sobre train. La evaluación sobre val ocurre una única
// vez al final de Run(), sobre el mejor individuo final, con lambda=0.
func NewEvolution(cfg Config, train, val *dataset.Dataset) (*Evolution, error) {
	// Normalizar Workers antes de validate: ≤ 0 → 1.
	if cfg.Workers <= 0 {
		cfg.Workers = 1
	}

	if err := cfg.validate(); err != nil {
		return nil, err
	}
	if train == nil || train.Len() == 0 {
		return nil, fmt.Errorf("genetic: training dataset is nil or empty")
	}
	// val == nil es válido (sin validación).
	// val con 0 puntos es error: distingue "omitir validación" de "dataset roto".
	if val != nil && val.Len() == 0 {
		return nil, fmt.Errorf("genetic: validation dataset must not be empty (pass nil to skip validation)")
	}
	gen, err := generator.New(cfg.GenConfig)
	if err != nil {
		return nil, fmt.Errorf("genetic: invalid generator config: %w", err)
	}
	return &Evolution{
		cfg: cfg,
		rng: rand.New(rand.NewSource(cfg.Seed)),
		ds:  train,
		val: val,
		gen: gen,
	}, nil
}

// updateStagnationCount devuelve el nuevo valor del contador de estancamiento.
//
// La comparación es estrictamente generación a generación: se evalúa si
// (prevFitness - newFitness) > delta para la generación actual.
// Varias mejoras consecutivas, cada una inferior a delta, incrementan el
// contador aunque su suma supere delta.
//
// Devuelve 0 si hubo mejora significativa; count+1 en caso contrario.
// Es una función pura sin efectos secundarios.
func updateStagnationCount(prevFitness, newFitness, delta float64, count int) int {
	if prevFitness-newFitness > delta {
		return 0
	}
	return count + 1
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
// Criterios de parada (Fase 14) — semántica OR, el primero en cumplirse termina:
//   - Generations: límite hard superior; siempre activo.
//   - TargetMSE: si TargetMSEEnabled=true, se evalúa en gen=0 (población inicial)
//     y en cada generación evolutiva. La evaluación en gen=0 permite terminar
//     sin ejecutar ninguna generación cuando la población inicial ya satisface
//     el objetivo; en ese caso GenerationsRun vale 0.
//   - Estancamiento: si StagnationLimit>0, el loop termina tras StagnationLimit
//     generaciones consecutivas sin mejora significativa en Fitness (medida por
//     StagnationDelta). El estancamiento no se evalúa en gen=0 porque no existe
//     baseline previo contra el cual comparar.
//
// Cuando TargetMSE y estancamiento se cumplen en la misma generación,
// TargetMSE tiene prioridad (se comprueba primero en el código).
//
// El globalBest nunca empeora: si gen N no produce nada mejor, se conserva
// el mejor encontrado hasta ese momento (garantizado por elitismo + tracking).
//
// OnGeneration se invoca:
//   - gen=0: con la población inicial, siempre antes de cualquier check de parada.
//   - gen=1..N: al finalizar cada generación procesada, incluida la que provoca
//     la parada, siempre antes de los checks de parada anticipada.
//
// Validation se evalúa exactamente una vez al finalizar Run(),
// independientemente del motivo de terminación.
// Nunca interviene en los criterios de parada.
func (e *Evolution) Run() (*Individual, error) {
	pop := e.InitPopulation()
	sortByFitness(pop)

	globalBest := cloneIndividual(pop[0])
	if e.cfg.OnGeneration != nil {
		e.cfg.OnGeneration(0, globalBest)
	}

	// Check de TargetMSE en gen=0: la población inicial ya está evaluada.
	// El estancamiento no aplica aquí porque no hay baseline previo.
	if e.cfg.TargetMSEEnabled && globalBest.Result.MSE <= e.cfg.TargetMSE {
		// GenerationsRun=0: ninguna generación evolutiva fue ejecutada.
		globalBest.GenerationsRun = 0
		return e.finalize(&globalBest)
	}

	stagnationCount := 0
	generationsRun := 0

	for gen := 1; gen <= e.cfg.Generations; gen++ {
		pop = e.NextGeneration(pop)
		sortByFitness(pop)

		prevFitness := globalBest.Result.Fitness
		if pop[0].Result.Fitness < globalBest.Result.Fitness {
			globalBest = cloneIndividual(pop[0])
		}

		// Actualizar contador de estancamiento (solo si el criterio está activo).
		if e.cfg.StagnationLimit > 0 {
			stagnationCount = updateStagnationCount(
				prevFitness, globalBest.Result.Fitness,
				e.cfg.StagnationDelta, stagnationCount,
			)
		}

		// Informar generación procesada — siempre antes de los checks de parada,
		// de modo que la generación que gatilla la terminación también se reporta.
		if e.cfg.OnGeneration != nil {
			e.cfg.OnGeneration(gen, globalBest)
		}

		generationsRun = gen

		// Checks de parada anticipada — semántica OR, TargetMSE primero.
		if e.cfg.TargetMSEEnabled && globalBest.Result.MSE <= e.cfg.TargetMSE {
			break
		}
		if e.cfg.StagnationLimit > 0 && stagnationCount >= e.cfg.StagnationLimit {
			break
		}
	}

	globalBest.GenerationsRun = generationsRun
	return e.finalize(&globalBest)
}

// finalize evalúa el dataset de validación sobre globalBest y retorna el resultado.
// Se llama siempre al final de Run(), independientemente del motivo de terminación.
// lambda=0: MSE puro sin penalización de complejidad (el spec reporta "Validation MSE").
// Esta evaluación no modifica el RNG ni los resultados de entrenamiento.
func (e *Evolution) finalize(globalBest *Individual) (*Individual, error) {
	if e.val != nil {
		valResult, err := fitness.Evaluate(globalBest.Expr, e.val, 0)
		if err != nil {
			return nil, fmt.Errorf("genetic: validation evaluation failed: %w", err)
		}
		globalBest.ValidationResult = &valResult
	}
	return globalBest, nil
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

	// Fase 1a — élites: se copian con sus resultados existentes (sin re-evaluar).
	// EliteCount() es la fuente única de verdad para este cálculo.
	elites := e.cfg.EliteCount()
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
