package dataset

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
)

// minPoints es el mínimo de puntos de datos aceptado en un dataset.
const minPoints = 2

// ParseCSV lee un dataset desde cualquier io.Reader.
// Espera una fila de encabezado "x,y" seguida de filas con dos valores float64.
// Rechaza NaN, Inf, valores no numéricos, columnas incorrectas y datasets vacíos.
// Exportada para facilitar el testing sin I/O de disco.
func ParseCSV(r io.Reader) (*Dataset, error) {
	cr := csv.NewReader(r)
	cr.FieldsPerRecord = 2    // exactamente 2 columnas en cada fila
	cr.TrimLeadingSpace = true

	// Leer y validar encabezado
	header, err := cr.Read()
	if err != nil {
		return nil, fmt.Errorf("dataset: no se pudo leer el encabezado: %w", err)
	}
	if !isValidHeader(header) {
		return nil, fmt.Errorf("dataset: encabezado inválido %v, se esperaba [x y]", header)
	}

	// Leer filas de datos
	var points []Point
	lineNum := 1 // encabezado fue la línea 1
	for {
		record, err := cr.Read()
		if errors.Is(err, io.EOF) {
			break
		}
		lineNum++
		if err != nil {
			return nil, fmt.Errorf("dataset: línea %d: %w", lineNum, err)
		}

		x, err := parseField(record[0], "x", lineNum)
		if err != nil {
			return nil, err
		}
		y, err := parseField(record[1], "y", lineNum)
		if err != nil {
			return nil, err
		}

		points = append(points, Point{X: x, Y: y})
	}

	if len(points) < minPoints {
		return nil, fmt.Errorf("dataset: se necesitan al menos %d puntos, se encontraron %d",
			minPoints, len(points))
	}

	return New(points), nil
}

// LoadCSV abre el archivo en path y llama a ParseCSV.
func LoadCSV(path string) (*Dataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("dataset: no se pudo abrir %q: %w", path, err)
	}
	defer f.Close()
	return ParseCSV(f)
}

// isValidHeader devuelve true si h tiene exactamente los campos "x" e "y"
// (insensible a mayúsculas y espacios circundantes).
func isValidHeader(h []string) bool {
	if len(h) != 2 {
		return false
	}
	return strings.ToLower(strings.TrimSpace(h[0])) == "x" &&
		strings.ToLower(strings.TrimSpace(h[1])) == "y"
}

// parseField parsea un campo CSV como float64 y rechaza NaN e Inf.
// strconv.ParseFloat acepta "NaN", "+Inf", "-Inf" sin error, por eso
// se verifica explícitamente después del parseo.
func parseField(s, name string, line int) (float64, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, fmt.Errorf("dataset: línea %d: valor vacío para %s", line, name)
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, fmt.Errorf("dataset: línea %d: valor no numérico para %s: %q", line, name, s)
	}
	if math.IsNaN(v) {
		return 0, fmt.Errorf("dataset: línea %d: valor NaN para %s", line, name)
	}
	if math.IsInf(v, 0) {
		return 0, fmt.Errorf("dataset: línea %d: valor Inf para %s", line, name)
	}
	return v, nil
}
