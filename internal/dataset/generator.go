package dataset

// Generate crea un Dataset evaluando fn en n valores de x equiespaciados
// en el intervalo cerrado [xMin, xMax].
//
// Los valores de x son deterministas: llamadas sucesivas producen el mismo resultado.
// El último punto se ancla a xMax para evitar la acumulación de error de float64
// que ocurre al sumar pasos repetidamente.
//
// Casos especiales:
//   - n <= 0 devuelve un dataset vacío.
//   - n == 1 devuelve un único punto en x = xMin.
func Generate(fn func(float64) float64, xMin, xMax float64, n int) *Dataset {
	if n <= 0 {
		return New(nil)
	}
	if n == 1 {
		return New([]Point{{X: xMin, Y: fn(xMin)}})
	}

	points := make([]Point, n)
	step := (xMax - xMin) / float64(n-1)
	for i := 0; i < n; i++ {
		var x float64
		if i == n-1 {
			x = xMax // anclado para evitar drift acumulado
		} else {
			x = xMin + float64(i)*step
		}
		points[i] = Point{X: x, Y: fn(x)}
	}
	return New(points)
}
