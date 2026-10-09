package genetic

import (
	"github.com/LautiSeverino/symbolic-regression/internal/expression"
	"github.com/LautiSeverino/symbolic-regression/internal/fitness"
)

// evalJob describe un trabajo de evaluación de fitness.
// index preserva la posición original del individuo para que los resultados
// puedan reensamblarse en orden determinista independientemente del scheduling.
type evalJob struct {
	index int
	expr  *expression.Node
}

// evalResult transporta el resultado de un worker de vuelta al coordinador.
type evalResult struct {
	index  int
	result fitness.FitnessResult
}

// evaluatePopulation evalúa un slice de expresiones usando e.cfg.Workers goroutines
// y devuelve los resultados en el mismo orden que las expresiones de entrada.
//
// Contrato de concurrencia:
//   - Los workers solo leen e.ds y e.cfg.Lambda, ambos inmutables post-construcción.
//   - No se comparte ningún estado mutable (sin RNG compartido, sin escrituras a e).
//   - fitness.Evaluate y expression.Evaluate son funciones puras.
//   - Por lo tanto, Workers=1 y Workers>1 producen resultados idénticos para las
//     mismas expresiones, y el race detector no detecta conflictos.
//
// Ciclo de vida:
//   - Los canales jobs y results son locales a esta llamada.
//   - close(jobs) señaliza a los workers que no hay más trabajo; sus range loops
//     terminan y las goroutines finalizan. No hay goroutine leaks.
//   - Si len(exprs) == 0 devuelve nil sin lanzar workers.
//
// Workers reales usados: min(e.cfg.Workers, len(exprs)).
// Lanzar más workers que jobs sería inútil; el ajuste evita goroutines ociosas.
func (e *Evolution) evaluatePopulation(exprs []*expression.Node) []fitness.FitnessResult {
	n := len(exprs)
	if n == 0 {
		return nil
	}

	numWorkers := e.cfg.Workers
	if numWorkers > n {
		numWorkers = n
	}

	// Canales completamente buffered: ningún sender se bloquea esperando un receiver.
	// Con capacidad n, todos los jobs pueden encolarse y todos los resultados pueden
	// depositarse antes de que el coordinador comience a leerlos.
	jobs := make(chan evalJob, n)
	results := make(chan evalResult, n)

	// Lanzar workers. Cada goroutine itera sobre jobs hasta que el canal se cierre.
	// No comparten estado mutable: evalExpr solo lee e.ds y e.cfg.Lambda.
	for w := 0; w < numWorkers; w++ {
		go func() {
			for job := range jobs {
				results <- evalResult{
					index:  job.index,
					result: e.evalExpr(job.expr),
				}
			}
		}()
	}

	// Enviar todos los trabajos. Los sends no bloquean (canal buffered con cap=n).
	for i, expr := range exprs {
		jobs <- evalJob{index: i, expr: expr}
	}
	// Cerrar el canal: los workers saldrán de su range loop cuando lo drenen.
	close(jobs)

	// Recolectar todos los resultados. El índice de cada resultado determina
	// su posición en out, garantizando orden determinista sin importar el
	// scheduling de goroutines.
	out := make([]fitness.FitnessResult, n)
	for range n {
		r := <-results
		out[r.index] = r.result
	}

	return out
}
