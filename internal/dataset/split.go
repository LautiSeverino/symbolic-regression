package dataset

import (
	"fmt"
	"math/rand"
)

// Split divide el dataset en subconjuntos de training y validation.
//
// trainRatio debe estar en el intervalo abierto (0, 1).
// La mezcla usa una fuente de aleatoriedad local sembrada con seed,
// por lo que el resultado es completamente reproducible con los mismos
// dataset, ratio y seed.
//
// Garantías:
//   - train.Len() + validation.Len() == ds.Len() (ningún punto se pierde ni duplica)
//   - Al menos un punto en cada subconjunto, incluso para datasets pequeños.
func Split(ds *Dataset, trainRatio float64, seed int64) (train, validation *Dataset, err error) {
	if ds == nil || len(ds.Points) == 0 {
		return nil, nil, fmt.Errorf("dataset: no se puede dividir un dataset nil o vacío")
	}
	if trainRatio <= 0 || trainRatio >= 1 {
		return nil, nil, fmt.Errorf("dataset: trainRatio debe estar en (0, 1), se recibió %v", trainRatio)
	}

	n := len(ds.Points)

	// Construir índices y mezclarlos con fuente local (no global) para reproducibilidad
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}
	rng := rand.New(rand.NewSource(seed)) //nolint:gosec // no es uso criptográfico
	rng.Shuffle(n, func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})

	// Calcular tamaño de training con guardas para datasets pequeños
	trainN := int(float64(n) * trainRatio)
	if trainN < 1 {
		trainN = 1 // mínimo 1 punto en training
	}
	if trainN >= n {
		trainN = n - 1 // mínimo 1 punto en validation
	}

	trainPoints := make([]Point, trainN)
	for i, idx := range indices[:trainN] {
		trainPoints[i] = ds.Points[idx]
	}

	valPoints := make([]Point, n-trainN)
	for i, idx := range indices[trainN:] {
		valPoints[i] = ds.Points[idx]
	}

	return New(trainPoints), New(valPoints), nil
}
