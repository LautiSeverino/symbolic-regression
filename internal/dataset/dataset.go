package dataset

// Point representa un único par de observación (x, y).
type Point struct {
	X float64
	Y float64
}

// Dataset contiene una colección de pares (x, y).
// Es la estructura que reciben las capas de fitness y evaluación.
type Dataset struct {
	Points []Point
}

// New crea un Dataset a partir de una slice de Points existente.
// La slice se referencia directamente, no se copia.
func New(points []Point) *Dataset {
	return &Dataset{Points: points}
}

// Len devuelve la cantidad de puntos. Es seguro llamarlo sobre un Dataset nil.
func (d *Dataset) Len() int {
	if d == nil {
		return 0
	}
	return len(d.Points)
}
