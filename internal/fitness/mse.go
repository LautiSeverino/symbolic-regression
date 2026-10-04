package fitness

import (
	"fmt"
	"math"

	"github.com/LautiSeverino/symbolic-regression/internal/dataset"
	"github.com/LautiSeverino/symbolic-regression/internal/expression"
)

// PenaltyPerInvalidPoint es la contribución al error cuadrático asignada a cada
// punto del dataset donde expression.Evaluate devuelve un error (división por cero,
// sqrt de negativo, ln de no-positivo, etc.).
//
// Estrategia: penalización finita, no +Inf.
// Usar un valor finito permite que la invalidad parcial sea informativa:
// una expresión que falla en 1 de 100 puntos obtiene un MSE mucho menor que
// una que falla en los 100. Con +Inf ambas serían indistinguibles.
//
// Valor 1e9: varios órdenes de magnitud por encima del MSE típico en el dominio
// esperado (x ∈ [-100, 100], y acotado), haciendo que las expresiones inválidas
// sean fuertemente no-atractivas sin perder el ordenamiento relativo.
const PenaltyPerInvalidPoint = 1e9

// FitnessResult contiene todos los componentes de la evaluación de fitness.
// Tener los cuatro valores juntos evita re-recorrer el dataset y permite
// al caller inspeccionar cada componente de forma independiente.
type FitnessResult struct {
	MSE          float64 // error cuadrático medio (PenaltyPerInvalidPoint por cada punto inválido)
	Complexity   int     // expression.Count(expr) — cantidad total de nodos
	Fitness      float64 // MSE + lambda * float64(Complexity)
	InvalidCount int     // puntos donde expression.Evaluate devolvió un error
}

// Evaluate calcula el fitness de expr sobre ds con el peso lambda.
//
// La fórmula de fitness combinado es:
//
//	fitness = MSE + lambda * complexity
//
// Pasar lambda=0 usa solo el MSE sin penalización de complejidad.
//
// Devuelve un error si ds es nil/vacío, o si el Fitness final resulta
// no-finito (NaN o ±Inf), lo que puede ocurrir cuando lambda es no-finito.
//
// Evaluate nunca produce panic: cualquier EvalError de expression.Evaluate
// queda registrado en FitnessResult.InvalidCount y reemplazado por
// PenaltyPerInvalidPoint en el cálculo del MSE.
func Evaluate(expr *expression.Node, ds *dataset.Dataset, lambda float64) (FitnessResult, error) {
	if ds == nil || ds.Len() == 0 {
		return FitnessResult{}, fmt.Errorf("fitness: dataset es nil o está vacío")
	}

	var sumSq float64
	var invalidCount int

	for _, p := range ds.Points {
		pred, err := expression.Evaluate(expr, p.X)
		if err != nil {
			// EvalError: expresión indefinida en este punto → penalizar
			sumSq += PenaltyPerInvalidPoint
			invalidCount++
			continue
		}

		diff := p.Y - pred
		sq := diff * diff
		if math.IsInf(sq, 0) {
			// Overflow en el error cuadrático (e.g., pred ≈ ±MaxFloat64 con y de signo contrario)
			// Se trata como punto inválido para evitar contaminar el MSE con Inf.
			sumSq += PenaltyPerInvalidPoint
			invalidCount++
			continue
		}
		sumSq += sq
	}

	n := float64(ds.Len())
	mse := sumSq / n
	complexity := expression.Count(expr)
	fitness := mse + lambda*float64(complexity)

	if math.IsNaN(fitness) || math.IsInf(fitness, 0) {
		return FitnessResult{}, fmt.Errorf(
			"fitness: resultado no-finito (MSE=%v, lambda=%v, complexity=%d)",
			mse, lambda, complexity,
		)
	}

	return FitnessResult{
		MSE:          mse,
		Complexity:   complexity,
		Fitness:      fitness,
		InvalidCount: invalidCount,
	}, nil
}
