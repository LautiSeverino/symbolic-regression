package genetic

import (
	"fmt"
	"math"

	"github.com/LautiSeverino/symbolic-regression/internal/expression"
	"github.com/LautiSeverino/symbolic-regression/internal/fitness"
	"github.com/LautiSeverino/symbolic-regression/internal/generator"
)

// Individual es un individuo de la población del algoritmo genético.
// ValidationResult es nil durante la evolución; solo Run() lo asigna
// cuando se proporciona un dataset de validación.
// GenerationsRun es el número de generaciones evolutivas ejecutadas (1..N).
// Vale 0 si la evolución terminó antes de ejecutar cualquier generación
// (p.ej. TargetMSE satisfecho en la población inicial). Solo Run() asigna
// este campo; vale 0 en todos los demás contextos.
type Individual struct {
	Expr             *expression.Node
	Result           fitness.FitnessResult  // métricas de entrenamiento
	ValidationResult *fitness.FitnessResult // nil si no hay val dataset; evaluado con lambda=0
	GenerationsRun   int                    // generaciones evolutivas ejecutadas; solo Run() lo asigna
}

// cloneIndividual devuelve una copia independiente del individuo.
// Si ValidationResult no es nil, se copia su valor (copia defensiva del struct).
// GenerationsRun se copia por valor.
func cloneIndividual(ind Individual) Individual {
	clone := Individual{
		Expr:           expression.Clone(ind.Expr),
		Result:         ind.Result,
		GenerationsRun: ind.GenerationsRun,
	}
	if ind.ValidationResult != nil {
		vr := *ind.ValidationResult // copia del valor, no del puntero
		clone.ValidationResult = &vr
	}
	return clone
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

	// --- Criterios de parada anticipada (Fase 14) ---

	// TargetMSEEnabled activa el criterio de MSE objetivo cuando es true.
	// Cuando es false, TargetMSE se ignora completamente. Default: false.
	// El zero value (false) garantiza compatibilidad con Config existentes que
	// omitan este campo: el criterio permanece deshabilitado.
	TargetMSEEnabled bool

	// TargetMSE es el umbral de MSE para la parada anticipada.
	// Solo tiene efecto cuando TargetMSEEnabled es true.
	// La evolución se detiene cuando globalBest.Result.MSE <= TargetMSE.
	// Debe ser >= 0 cuando TargetMSEEnabled es true. No debe ser NaN en ningún caso.
	// Default: 0 (irrelevante cuando TargetMSEEnabled es false).
	TargetMSE float64

	// StagnationLimit activa la parada por estancamiento si globalBest no mejora
	// durante StagnationLimit generaciones consecutivas. 0 deshabilita el criterio.
	// El zero value (0) garantiza compatibilidad hacia atrás. Default: 0.
	StagnationLimit int

	// StagnationDelta define la mejora mínima en Fitness necesaria para resetear
	// el contador de estancamiento. La comparación es estrictamente generación
	// a generación: se evalúa si (prevFitness - newFitness) > StagnationDelta
	// para cada generación individual.
	// Varias mejoras consecutivas, cada una inferior a StagnationDelta, incrementan
	// el contador aunque su suma supere StagnationDelta.
	// 0 significa mejora estricta (cualquier descenso en Fitness resetea). Default: 0.
	StagnationDelta float64

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
	// Se invoca siempre antes de los checks de parada anticipada, de modo que
	// la generación que gatilla la terminación también es reportada.
	OnGeneration func(gen int, best Individual)
}

// DefaultConfig devuelve una configuración de desarrollo con parámetros pequeños.
// No es un perfil FAST/BALANCED — los perfiles pertenecen a Fase 11.
// Los criterios de parada anticipada están deshabilitados por defecto.
func DefaultConfig() Config {
	return Config{
		PopulationSize:   100,
		Generations:      50,
		TournamentSize:   3,
		ElitismRate:      0.05,
		MutationRate:     0.10,
		CrossoverRate:    0.80,
		Lambda:           0.01,
		Seed:             0,
		Workers:          1,
		TargetMSEEnabled: false,
		TargetMSE:        0,
		StagnationLimit:  0,
		StagnationDelta:  0,
		GenConfig:        generator.Default(),
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
	// Validaciones de criterios de parada anticipada (Fase 14).
	// NaN se rechaza siempre, independientemente de TargetMSEEnabled,
	// porque un NaN en el struct es casi siempre un bug del caller.
	if math.IsNaN(cfg.TargetMSE) {
		return fmt.Errorf("genetic: TargetMSE must not be NaN")
	}
	if cfg.TargetMSEEnabled {
		if cfg.TargetMSE < 0 {
			return fmt.Errorf("genetic: TargetMSE must be ≥ 0 when TargetMSEEnabled is true, got %v", cfg.TargetMSE)
		}
		if math.IsInf(cfg.TargetMSE, 1) {
			return fmt.Errorf("genetic: TargetMSE must not be +Inf when TargetMSEEnabled is true")
		}
	}
	if cfg.StagnationLimit < 0 {
		return fmt.Errorf("genetic: StagnationLimit must be ≥ 0, got %d", cfg.StagnationLimit)
	}
	if math.IsNaN(cfg.StagnationDelta) || math.IsInf(cfg.StagnationDelta, 0) {
		return fmt.Errorf("genetic: StagnationDelta must be finite, got %v", cfg.StagnationDelta)
	}
	if cfg.StagnationDelta < 0 {
		return fmt.Errorf("genetic: StagnationDelta must be ≥ 0, got %v", cfg.StagnationDelta)
	}
	// Workers ya fue normalizado en NewEvolution; aquí cfg.Workers >= 1.
	return nil
}

// EliteCount devuelve cuántos individuos se conservan por elitismo.
// Es la fuente única de verdad para este cálculo; tanto NextGeneration
// como main.go deben llamar a este método para evitar duplicar la fórmula.
func (cfg Config) EliteCount() int {
	n := int(float64(cfg.PopulationSize) * cfg.ElitismRate)
	if n < 0 {
		n = 0
	}
	if n >= cfg.PopulationSize {
		n = cfg.PopulationSize - 1
	}
	return n
}
