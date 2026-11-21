// Package main: cli.go — parsing y aplicación de overrides de la CLI.
//
// Contiene:
//   - cliOptions: estructura con los valores de todos los flags resueltos.
//   - parseCLI: interpreta os.Args[1:] (o cualquier slice de strings) y
//     devuelve cliOptions con el conjunto de flags explícitamente especificados.
//   - applyConfigOverrides: aplica los overrides explícitos sobre la Config
//     base cargada por config.Parse().
package main

import (
	"flag"
	"fmt"
	"io"

	"github.com/LautiSeverino/symbolic-regression/internal/genetic"
)

// defaultDataPath es el dataset utilizado cuando el usuario no especifica ninguno.
const defaultDataPath = "datasets/linear.csv"

// cliOptions contiene todos los valores de los flags resueltos y el conjunto
// de flags que el usuario especificó explícitamente.
//
// Explicit indica qué flags aparecieron en la línea de comandos.
// No indica si el valor es válido para el algoritmo genético; la validación
// de rangos ocurre en genetic.NewEvolution → Config.validate().
type cliOptions struct {
	DataPath    string
	Profile     string
	Seed        int64
	Population  int
	Generations int
	Workers     int
	Depth       int
	Nodes       int
	Explicit    map[string]bool
}

// parseCLI interpreta args (normalmente os.Args[1:]) y devuelve las opciones
// resueltas o un error.
//
// out controla dónde escribe el flag set los mensajes de error y la ayuda.
// Pasar nil equivale a os.Stderr (comportamiento por defecto del flag package).
// En tests se puede pasar io.Discard para silenciar la salida.
//
// Reglas de resolución de DataPath (en orden de prioridad):
//  1. --data explícito: se usa ese valor.
//  2. --data ausente + un argumento posicional: se usa el argumento posicional
//     (compatibilidad con la CLI anterior a la Fase 15).
//  3. Ninguno: se usa defaultDataPath ("datasets/linear.csv").
//
// Errores:
//   - --data y argumento posicional simultáneos → ambigüedad.
//   - Más de un argumento posicional.
//   - flag.ErrHelp si el usuario pasa --help / -h.
//   - Cualquier otro error de parsing de flags (flag desconocido, valor inválido).
func parseCLI(args []string, out io.Writer) (cliOptions, error) {
	fs := flag.NewFlagSet("symbolic-regression", flag.ContinueOnError)
	fs.SetOutput(out) // nil → os.Stderr (flag.FlagSet default via Output())

	dataFlag := fs.String("data", defaultDataPath, "ruta al archivo CSV del dataset")
	profileFlag := fs.String("profile", "balanced", "perfil de ejecución: fast | balanced | intensive | benchmark")
	seedFlag := fs.Int64("seed", 42, "semilla para reproducibilidad (misma seed + perfil + dataset = mismo resultado)")
	popFlag := fs.Int("population", 0, "sobrescribe el tamaño de población del perfil (≥ 2)")
	genFlag := fs.Int("generations", 0, "sobrescribe el número de generaciones del perfil (≥ 1)")
	workFlag := fs.Int("workers", 0, "sobrescribe la cantidad de workers del perfil (≥ 1; <= 0 se normaliza a 1 en la evolución)")
	depthFlag := fs.Int("depth", 0, "sobrescribe la profundidad máxima del árbol del perfil (≥ 1)")
	nodesFlag := fs.Int("nodes", 0, "sobrescribe la cantidad máxima de nodos del árbol del perfil (≥ 1)")

	fs.Usage = func() {
		fmt.Fprintf(fs.Output(),
			"Uso: symbolic-regression [--data DATASET] [--profile PERFIL] [--seed N] [opciones] [DATASET]\n\n"+
				"Opciones:\n")
		fs.PrintDefaults()
		fmt.Fprintf(fs.Output(),
			"\nEjemplos:\n"+
				"  symbolic-regression --data datasets/quadratic.csv --profile balanced --seed 42\n"+
				"  symbolic-regression --population 2000 --generations 500 --workers 8 --seed 42\n"+
				"  symbolic-regression datasets/linear.csv\n")
	}

	if err := fs.Parse(args); err != nil {
		return cliOptions{}, err
	}

	// Detectar flags explícitamente especificados.
	explicit := make(map[string]bool)
	fs.Visit(func(f *flag.Flag) {
		explicit[f.Name] = true
	})

	// Validar argumentos posicionales.
	positional := fs.Args()
	if len(positional) > 1 {
		return cliOptions{}, fmt.Errorf(
			"demasiados argumentos posicionales: se esperaba 1 como máximo, se recibieron %d",
			len(positional),
		)
	}

	// Resolver DataPath con la jerarquía de prioridad definida.
	dataPath := defaultDataPath
	if explicit["data"] {
		if len(positional) > 0 {
			return cliOptions{}, fmt.Errorf(
				"uso ambiguo: --data y argumento posicional no pueden usarse simultáneamente",
			)
		}
		dataPath = *dataFlag
	} else if len(positional) == 1 {
		dataPath = positional[0]
	}

	return cliOptions{
		DataPath:    dataPath,
		Profile:     *profileFlag,
		Seed:        *seedFlag,
		Population:  *popFlag,
		Generations: *genFlag,
		Workers:     *workFlag,
		Depth:       *depthFlag,
		Nodes:       *nodesFlag,
		Explicit:    explicit,
	}, nil
}

// applyConfigOverrides aplica los flags explícitos de opts sobre cfg y
// devuelve la configuración resultante.
//
// Solo modifica los campos correspondientes a flags que el usuario especificó
// explícitamente. Los campos no explícitos conservan el valor del perfil base.
//
// Notas:
//   - Workers: si el usuario pasa --workers 0, cfg.Workers quedará en 0.
//     genetic.NewEvolution normalizará ese valor a 1 de acuerdo con la semántica
//     establecida en la Fase 10.
//   - La validación de rangos (PopulationSize ≥ 2, Generations ≥ 1, etc.)
//     es responsabilidad de genetic.NewEvolution → Config.validate().
func applyConfigOverrides(cfg genetic.Config, opts cliOptions) genetic.Config {
	if opts.Explicit["population"] {
		cfg.PopulationSize = opts.Population
	}
	if opts.Explicit["generations"] {
		cfg.Generations = opts.Generations
	}
	if opts.Explicit["workers"] {
		cfg.Workers = opts.Workers
	}
	if opts.Explicit["depth"] {
		cfg.GenConfig.MaxDepth = opts.Depth
	}
	if opts.Explicit["nodes"] {
		cfg.GenConfig.MaxNodes = opts.Nodes
	}
	return cfg
}

// normalizeConfig aplica las normalizaciones de rango que genetic.NewEvolution
// también aplica internamente. Llamarla antes del bloque de impresión en main
// garantiza que la configuración mostrada coincida con los valores reales usados
// durante la evolución.
//
// Normalizaciones aplicadas:
//   - Workers <= 0 → 1 (misma semántica que genetic.NewEvolution, Fase 10).
func normalizeConfig(cfg genetic.Config) genetic.Config {
	if cfg.Workers <= 0 {
		cfg.Workers = 1
	}
	return cfg
}
