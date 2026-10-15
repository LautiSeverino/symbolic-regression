// Package config define los perfiles de ejecución del sistema de regresión simbólica.
//
// Un perfil es un preset de presupuesto computacional: determina el tamaño de
// la población, las generaciones, la profundidad máxima de los árboles y la
// cantidad de workers. No cambia el algoritmo genético ni el lenguaje matemático
// disponible; todos los perfiles utilizan el mismo conjunto de operadores y los
// mismos parámetros evolutivos (torneo, elitismo, mutación, crossover, lambda).
//
// La Seed, tanto del GA como del generador, queda en 0 en cada perfil y debe
// ser asignada por el caller antes de crear una Evolution.
package config

import (
	"fmt"
	"strings"

	"github.com/LautiSeverino/symbolic-regression/internal/expression"
	"github.com/LautiSeverino/symbolic-regression/internal/generator"
	"github.com/LautiSeverino/symbolic-regression/internal/genetic"
)

// ProfileName identifica un perfil de ejecución.
type ProfileName string

const (
	// ProfileFAST está pensado para el ciclo de desarrollo rápido.
	// Debug, tests manuales y verificación de cambios. Ejecuta en segundos.
	ProfileFAST ProfileName = "fast"

	// ProfileBALANCED es el perfil principal para ejecuciones normales.
	// Datasets de tamaño medio y experimentos habituales. Es el valor por defecto.
	ProfileBALANCED ProfileName = "balanced"

	// ProfileINTENSIVE es para búsquedas agresivas en problemas difíciles donde
	// BALANCED no converge. No es el perfil por defecto. Requiere tiempo
	// de ejecución significativo en el hardware objetivo.
	ProfileINTENSIVE ProfileName = "intensive"

	// ProfileBENCHMARK está destinado a medición y comparación de rendimiento.
	// Sus parámetros son idénticos a BALANCED; la distinción es semántica:
	// señala que la ejecución está destinada a benchmarks. Los benchmarks
	// formales de Fase 16 usan este perfil como base fija, variando exactamente
	// un parámetro a la vez (workers, población, etc.) para aislar su efecto.
	// Siempre debe usarse con --seed explícito para que los resultados sean
	// reproducibles y comparables entre ejecuciones.
	ProfileBENCHMARK ProfileName = "benchmark"
)

// Parse devuelve la genetic.Config correspondiente al nombre de perfil dado.
// El nombre es insensible a mayúsculas/minúsculas y a espacios circundantes.
//
// La Seed (tanto del GA como del generador) queda en 0; el caller debe
// asignarla antes de pasar la configuración a genetic.NewEvolution.
func Parse(name string) (genetic.Config, error) {
	switch ProfileName(strings.ToLower(strings.TrimSpace(name))) {
	case ProfileFAST:
		return FASTConfig(), nil
	case ProfileBALANCED:
		return BALANCEDConfig(), nil
	case ProfileINTENSIVE:
		return INTENSIVEConfig(), nil
	case ProfileBENCHMARK:
		return BENCHMARKConfig(), nil
	default:
		return genetic.Config{}, fmt.Errorf(
			"config: perfil desconocido %q (opciones: fast, balanced, intensive, benchmark)",
			name,
		)
	}
}

// FASTConfig devuelve la configuración del perfil FAST.
//
//	Population:  250  |  Generations:   50
//	Depth:         5  |  Max nodes:     40  |  Workers: 4
func FASTConfig() genetic.Config {
	return genetic.Config{
		PopulationSize: 250,
		Generations:    50,
		TournamentSize: 3,
		ElitismRate:    0.05,
		MutationRate:   0.10,
		CrossoverRate:  0.80,
		Lambda:         0.01,
		Workers:        4,
		GenConfig:      standardGenConfig(5, 40),
	}
}

// BALANCEDConfig devuelve la configuración del perfil BALANCED.
//
//	Population: 1.000  |  Generations:  300
//	Depth:          7  |  Max nodes:     75  |  Workers: 8
func BALANCEDConfig() genetic.Config {
	return genetic.Config{
		PopulationSize: 1_000,
		Generations:    300,
		TournamentSize: 3,
		ElitismRate:    0.05,
		MutationRate:   0.10,
		CrossoverRate:  0.80,
		Lambda:         0.01,
		Workers:        8,
		GenConfig:      standardGenConfig(7, 75),
	}
}

// INTENSIVEConfig devuelve la configuración del perfil INTENSIVE.
//
//	Population: 5.000  |  Generations: 1.000
//	Depth:         10  |  Max nodes:    100  |  Workers: 8
func INTENSIVEConfig() genetic.Config {
	return genetic.Config{
		PopulationSize: 5_000,
		Generations:    1_000,
		TournamentSize: 3,
		ElitismRate:    0.05,
		MutationRate:   0.10,
		CrossoverRate:  0.80,
		Lambda:         0.01,
		Workers:        8,
		GenConfig:      standardGenConfig(10, 100),
	}
}

// BENCHMARKConfig devuelve la configuración del perfil BENCHMARK.
// Sus parámetros son idénticos a BALANCED; la distinción es semántica.
// Ver la documentación de ProfileBENCHMARK para el uso correcto.
func BENCHMARKConfig() genetic.Config {
	return BALANCEDConfig()
}

// standardGenConfig construye la generator.Config estándar para los valores
// de maxDepth y maxNodes dados. El resto de los parámetros —operadores,
// constantes, probabilidades— es idéntico en todos los perfiles.
//
// Operadores binarios: +, -, *, /  (NodePow excluido; disponible como opt-in).
// Operadores unarios:  sqrt, ln, abs, exp, sin, cos  (todos los de Fase 8).
//
// Cada llamada crea slices independientes para evitar aliasing entre perfiles.
func standardGenConfig(maxDepth, maxNodes int) generator.Config {
	return generator.Config{
		MaxDepth:            maxDepth,
		MaxNodes:            maxNodes,
		AllowedBinaryOps:    standardBinaryOps(),
		AllowedUnaryOps:     standardUnaryOps(),
		ConstantMin:         -10,
		ConstantMax:         10,
		VariableProbability: 0.5,
		TerminalProbability: 0.3,
		PowExponentMin:      -4,
		PowExponentMax:      4,
		// Seed: 0 — el caller asigna cfg.GenConfig.Seed = seed+1 antes de NewEvolution.
	}
}

// standardBinaryOps devuelve el conjunto estándar de operadores binarios.
// Crea un nuevo slice en cada llamada para evitar mutaciones compartidas.
func standardBinaryOps() []expression.NodeType {
	return []expression.NodeType{
		expression.NodeAdd,
		expression.NodeSub,
		expression.NodeMul,
		expression.NodeDiv,
	}
}

// standardUnaryOps devuelve el conjunto estándar de funciones unarias.
// Incluye todos los operadores implementados hasta Fase 8.
// Crea un nuevo slice en cada llamada para evitar mutaciones compartidas.
func standardUnaryOps() []expression.NodeType {
	return []expression.NodeType{
		expression.NodeSqrt,
		expression.NodeLn,
		expression.NodeAbs,
		expression.NodeExp,
		expression.NodeSin,
		expression.NodeCos,
	}
}
