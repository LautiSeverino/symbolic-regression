package genetic

import (
	"fmt"

	"github.com/LautiSeverino/symbolic-regression/internal/expression"
	"github.com/LautiSeverino/symbolic-regression/internal/fitness"
	"github.com/LautiSeverino/symbolic-regression/internal/generator"
)

// Individual es un individuo de la población del algoritmo genético.
// Almacena la expresión candidata y el resultado completo del fitness.
type Individual struct {
	Expr   *expression.Node
	Result fitness.FitnessResult // contiene MSE, Complexity, Fitness, InvalidCount
}

// cloneIndividual devuelve una copia independiente del individuo.
// La expresión queda clonada para evitar aliasing con la población.
func cloneIndividual(ind Individual) Individual {
	return Individual{
		Expr:   expression.Clone(ind.Expr),
		Result: ind.Result,
	}
}

// Config contiene todos los parámetros del algoritmo genético.
type Config struct {
	PopulationSize int     // cantidad de individuos por generación (≥ 2)
	Generations    int     // rondas de evolución después de la población inicial (≥ 1)
	TournamentSize int     // participantes por torneo (≥ 2, ≤ PopulationSize)
	ElitismRate    float64 // fracción de la población que sobrevive intacta [0, 1)
	MutationRate   float64 // probabilidad de mutar cada hijo [0, 1]
	CrossoverRate  float64 // probabilidad de crossover (vs copia directa) [0, 1]
	Lambda         float64 // peso de la penalización de complejidad en el fitness
	Seed           int64   // semilla del RNG del GA; Seed+1 se usa para el generador

	// Workers es el número de goroutines del worker pool usado para evaluar fitness.
	//
	// Workers = 1  → evaluación secuencial (un único worker; comportamiento equivalente
	//                a la implementación anterior sin pool).
	// Workers > 1  → evaluación paralela; los workers comparten solo lectura de e.ds.
	// Workers <= 0 → normalizado a 1 automáticamente en NewEvolution.
	//
	// La evaluación de fitness es la única sección paralelizada. La generación
	// de descendencia (selección, crossover, mutación) permanece secuencial para
	// preservar el determinismo del RNG: misma seed + misma config → mismo resultado,
	// independientemente del número de workers.
	Workers int

	// GenConfig controla la generación de árboles aleatorios
	// (para la población inicial y para la mutación de subárboles).
	GenConfig generator.Config

	// OnGeneration es un callback opcional invocado al finalizar cada generación.
	// gen=0 corresponde a la población inicial; gen=1..Generations son las generaciones evolucionadas.
	// Se puede usar para registrar el progreso sin modificar el algoritmo.
	OnGeneration func(gen int, best Individual)
}

// DefaultConfig devuelve una configuración de desarrollo con parámetros pequeños.
// No es un perfil FAST/BALANCED — los perfiles pertenecen a Fase 11.
func DefaultConfig() Config {
	return Config{
		PopulationSize: 100,
		Generations:    50,
		TournamentSize: 3,
		ElitismRate:    0.05,
		MutationRate:   0.10,
		CrossoverRate:  0.80,
		Lambda:         0.01,
		Seed:           0,
		Workers:        1,
		GenConfig:      generator.Default(),
	}
}

// validate verifica que cfg tiene valores coherentes.
// Workers no se valida aquí; la normalización (≤ 0 → 1) ocurre en NewEvolution.
func (cfg Config) validate() error {
	if cfg.PopulationSize < 2 {
		return fmt.Errorf("genetic: PopulationSize must be ≥ 2, got %d", cfg.PopulationSize)
	}
	if cfg.Generations < 1 {
		return fmt.Errorf("genetic: Generations must be ≥ 1, got %d", cfg.Generations)
	}
	if cfg.TournamentSize < 2 {
		return fmt.Errorf("genetic: TournamentSize must be ≥ 2, got %d", cfg.TournamentSize)
	}
	if cfg.TournamentSize > cfg.PopulationSize {
		return fmt.Errorf("genetic: TournamentSize (%d) > PopulationSize (%d)",
			cfg.TournamentSize, cfg.PopulationSize)
	}
	if cfg.ElitismRate < 0 || cfg.ElitismRate >= 1 {
		return fmt.Errorf("genetic: ElitismRate must be in [0, 1), got %v", cfg.ElitismRate)
	}
	if cfg.MutationRate < 0 || cfg.MutationRate > 1 {
		return fmt.Errorf("genetic: MutationRate must be in [0, 1], got %v", cfg.MutationRate)
	}
	if cfg.CrossoverRate < 0 || cfg.CrossoverRate > 1 {
		return fmt.Errorf("genetic: CrossoverRate must be in [0, 1], got %v", cfg.CrossoverRate)
	}
	// Workers ya fue normalizado en NewEvolution; aquí cfg.Workers >= 1.
	return nil
}

// eliteCount devuelve cuántos individuos se conservan por elitismo.
func (cfg Config) eliteCount() int {
	n := int(float64(cfg.PopulationSize) * cfg.ElitismRate)
	if n < 0 {
		n = 0
	}
	if n >= cfg.PopulationSize {
		n = cfg.PopulationSize - 1
	}
	return n
}
