package dataset_test

import (
	"strings"
	"testing"

	"github.com/LautiSeverino/symbolic-regression/internal/dataset"
)

// ─── Helpers ──────────────────────────────────────────────────────────────

// makeDataset crea un dataset determinista de n puntos con f(x) = 2*x + 1.
func makeDataset(n int) *dataset.Dataset {
	points := make([]dataset.Point, n)
	for i := range points {
		x := float64(i + 1)
		points[i] = dataset.Point{X: x, Y: 2*x + 1}
	}
	return dataset.New(points)
}

// ─── Dataset.Len ──────────────────────────────────────────────────────────

func TestDataset_Len(t *testing.T) {
	var nilDs *dataset.Dataset
	if nilDs.Len() != 0 {
		t.Error("nil.Len() debe ser 0")
	}
	if dataset.New(nil).Len() != 0 {
		t.Error("New(nil).Len() debe ser 0")
	}
	ds := makeDataset(5)
	if ds.Len() != 5 {
		t.Errorf("Len() = %d, want 5", ds.Len())
	}
}

// ─── ParseCSV — casos válidos ──────────────────────────────────────────────

func TestParseCSV_Valid(t *testing.T) {
	input := "x,y\n1,3\n2,5\n3,7\n"
	ds, err := dataset.ParseCSV(strings.NewReader(input))
	if err != nil {
		t.Fatalf("error inesperado: %v", err)
	}
	if ds.Len() != 3 {
		t.Errorf("Len() = %d, want 3", ds.Len())
	}
	if ds.Points[0].X != 1 || ds.Points[0].Y != 3 {
		t.Errorf("points[0] = %v, want {1 3}", ds.Points[0])
	}
	if ds.Points[2].X != 3 || ds.Points[2].Y != 7 {
		t.Errorf("points[2] = %v, want {3 7}", ds.Points[2])
	}
}

// TestParseCSV_CanonicalDataset verifica el dataset canónico del spec: f(x) = 2*x + 1.
func TestParseCSV_CanonicalDataset(t *testing.T) {
	input := "x,y\n1,3\n2,5\n3,7\n4,9\n5,11\n"
	ds, err := dataset.ParseCSV(strings.NewReader(input))
	if err != nil {
		t.Fatalf("error inesperado: %v", err)
	}
	if ds.Len() != 5 {
		t.Errorf("Len() = %d, want 5", ds.Len())
	}
	for i, p := range ds.Points {
		want := 2*p.X + 1
		if p.Y != want {
			t.Errorf("points[%d]: Y=%v, want %v", i, p.Y, want)
		}
	}
}

func TestParseCSV_FloatValues(t *testing.T) {
	input := "x,y\n0.5,1.5\n1.5,2.5\n"
	ds, err := dataset.ParseCSV(strings.NewReader(input))
	if err != nil {
		t.Fatalf("error inesperado: %v", err)
	}
	if ds.Points[0].X != 0.5 || ds.Points[0].Y != 1.5 {
		t.Errorf("points[0] = %v, want {0.5 1.5}", ds.Points[0])
	}
}

func TestParseCSV_NegativeValues(t *testing.T) {
	input := "x,y\n-3,5\n-1,3\n"
	ds, err := dataset.ParseCSV(strings.NewReader(input))
	if err != nil {
		t.Fatalf("error inesperado: %v", err)
	}
	if ds.Points[0].X != -3 || ds.Points[0].Y != 5 {
		t.Errorf("points[0] = %v, want {-3 5}", ds.Points[0])
	}
}

func TestParseCSV_HeaderCaseInsensitive(t *testing.T) {
	// El encabezado es case-insensitive
	for _, header := range []string{"X,Y", "x,y", "X,y", "x,Y"} {
		input := header + "\n1,3\n2,5\n"
		_, err := dataset.ParseCSV(strings.NewReader(input))
		if err != nil {
			t.Errorf("header %q: error inesperado: %v", header, err)
		}
	}
}

// ─── ParseCSV — errores de encabezado ─────────────────────────────────────

func TestParseCSV_InvalidHeader(t *testing.T) {
	cases := []struct {
		name  string
		input string
	}{
		{"columnas incorrectas", "a,b\n1,2\n3,4\n"},
		{"columna vacía", ",\n1,2\n3,4\n"},
		{"primera columna incorrecta", "z,y\n1,2\n3,4\n"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := dataset.ParseCSV(strings.NewReader(tc.input))
			if err == nil {
				t.Errorf("input %q: se esperaba error por encabezado inválido, se obtuvo nil", tc.input)
			}
		})
	}
}

func TestParseCSV_WrongColumnCount(t *testing.T) {
	cases := []struct {
		name  string
		input string
	}{
		// El encabezado tiene solo 1 columna → csv.ErrFieldCount
		{"1 columna en header", "x\n1\n2\n"},
		// Fila de datos con 3 columnas → csv.ErrFieldCount
		{"3 columnas en datos", "x,y\n1,2,3\n4,5,6\n"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := dataset.ParseCSV(strings.NewReader(tc.input))
			if err == nil {
				t.Errorf("input %q: se esperaba error por cantidad de columnas incorrecta", tc.input)
			}
		})
	}
}

// ─── ParseCSV — errores de valores ────────────────────────────────────────

func TestParseCSV_NonNumeric(t *testing.T) {
	cases := []struct {
		name  string
		input string
	}{
		{"x no numérico", "x,y\nfoo,3\n2,5\n"},
		{"y no numérico", "x,y\n1,bar\n2,5\n"},
		{"ambos no numéricos", "x,y\none,two\nthree,four\n"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := dataset.ParseCSV(strings.NewReader(tc.input))
			if err == nil {
				t.Errorf("input %q: se esperaba error por valor no numérico", tc.input)
			}
		})
	}
}

func TestParseCSV_NaN(t *testing.T) {
	// strconv.ParseFloat acepta "NaN" sin error → debe rechazarse explícitamente
	cases := []struct {
		name  string
		input string
	}{
		{"NaN en x", "x,y\nNaN,3\n2,5\n"},
		{"NaN en y", "x,y\n1,NaN\n2,5\n"},
		{"nan minúscula", "x,y\nnan,3\n2,5\n"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := dataset.ParseCSV(strings.NewReader(tc.input))
			if err == nil {
				t.Errorf("input %q: se esperaba error por NaN", tc.input)
			}
		})
	}
}

func TestParseCSV_Inf(t *testing.T) {
	// strconv.ParseFloat acepta "+Inf", "-Inf", "Inf" sin error → rechazarlos
	cases := []struct {
		name  string
		input string
	}{
		{"+Inf en x", "x,y\n+Inf,3\n2,5\n"},
		{"-Inf en x", "x,y\n-Inf,3\n2,5\n"},
		{"Inf en x", "x,y\nInf,3\n2,5\n"},
		{"Inf en y", "x,y\n1,Inf\n2,5\n"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := dataset.ParseCSV(strings.NewReader(tc.input))
			if err == nil {
				t.Errorf("input %q: se esperaba error por Inf", tc.input)
			}
		})
	}
}

// ─── ParseCSV — datasets insuficientes ────────────────────────────────────

func TestParseCSV_TooFewPoints(t *testing.T) {
	cases := []struct {
		name  string
		input string
	}{
		{"solo encabezado", "x,y\n"},
		{"un solo punto", "x,y\n1,3\n"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := dataset.ParseCSV(strings.NewReader(tc.input))
			if err == nil {
				t.Errorf("input %q: se esperaba error por puntos insuficientes", tc.input)
			}
		})
	}
}

func TestParseCSV_EmptyInput(t *testing.T) {
	_, err := dataset.ParseCSV(strings.NewReader(""))
	if err == nil {
		t.Error("entrada vacía: se esperaba error, se obtuvo nil")
	}
}

// ─── LoadCSV ───────────────────────────────────────────────────────────────

func TestLoadCSV_NonExistentFile(t *testing.T) {
	_, err := dataset.LoadCSV("/nonexistent/path/dataset.csv")
	if err == nil {
		t.Error("se esperaba error para archivo inexistente, se obtuvo nil")
	}
}

// ─── Split ─────────────────────────────────────────────────────────────────

func TestSplit_DefaultRatio(t *testing.T) {
	// 10 puntos con 0.8: trainN = int(10*0.8) = 8, valN = 2
	ds := makeDataset(10)
	train, val, err := dataset.Split(ds, 0.8, 42)
	if err != nil {
		t.Fatalf("error inesperado: %v", err)
	}
	if train.Len() != 8 {
		t.Errorf("train.Len() = %d, want 8", train.Len())
	}
	if val.Len() != 2 {
		t.Errorf("val.Len() = %d, want 2", val.Len())
	}
}

func TestSplit_TotalPreserved(t *testing.T) {
	// La suma de puntos de train y validation debe igualar el total original.
	ds := makeDataset(10)
	train, val, err := dataset.Split(ds, 0.8, 42)
	if err != nil {
		t.Fatalf("error inesperado: %v", err)
	}
	if train.Len()+val.Len() != ds.Len() {
		t.Errorf("train(%d) + val(%d) != total(%d)", train.Len(), val.Len(), ds.Len())
	}
}

func TestSplit_AllPointsFromOriginal(t *testing.T) {
	// Todos los puntos del resultado deben provenir del dataset original.
	// Ningún punto debe perderse ni aparecer duplicado.
	n := 10
	ds := makeDataset(n)
	train, val, err := dataset.Split(ds, 0.8, 42)
	if err != nil {
		t.Fatalf("error inesperado: %v", err)
	}

	// Construir mapa de X originales (valores únicos 1..n)
	origXs := make(map[float64]struct{}, n)
	for _, p := range ds.Points {
		origXs[p.X] = struct{}{}
	}

	// Verificar que cada punto de train está en el original
	for _, p := range train.Points {
		if _, ok := origXs[p.X]; !ok {
			t.Errorf("train contiene X=%v que no está en el dataset original", p.X)
		}
	}
	// Verificar que cada punto de val está en el original
	for _, p := range val.Points {
		if _, ok := origXs[p.X]; !ok {
			t.Errorf("val contiene X=%v que no está en el dataset original", p.X)
		}
	}

	// Verificar que la unión cubre todos los originales (ningún punto perdido)
	splitXs := make(map[float64]struct{}, n)
	for _, p := range train.Points {
		splitXs[p.X] = struct{}{}
	}
	for _, p := range val.Points {
		splitXs[p.X] = struct{}{}
	}
	if len(splitXs) != n {
		t.Errorf("se esperaban %d X únicos en train∪val, se encontraron %d", n, len(splitXs))
	}
}

func TestSplit_Reproducible(t *testing.T) {
	// La misma seed debe producir exactamente el mismo split.
	ds := makeDataset(20)
	train1, val1, err := dataset.Split(ds, 0.8, 42)
	if err != nil {
		t.Fatalf("primera llamada: error inesperado: %v", err)
	}
	train2, val2, err := dataset.Split(ds, 0.8, 42)
	if err != nil {
		t.Fatalf("segunda llamada: error inesperado: %v", err)
	}

	if train1.Len() != train2.Len() {
		t.Fatalf("tamaños de train distintos: %d vs %d", train1.Len(), train2.Len())
	}
	for i := range train1.Points {
		if train1.Points[i] != train2.Points[i] {
			t.Errorf("train[%d]: %v != %v", i, train1.Points[i], train2.Points[i])
		}
	}
	for i := range val1.Points {
		if val1.Points[i] != val2.Points[i] {
			t.Errorf("val[%d]: %v != %v", i, val1.Points[i], val2.Points[i])
		}
	}
}

func TestSplit_DifferentSeeds(t *testing.T) {
	// Seeds distintas deben producir shuffles distintos.
	// Con n=20, P(mismo shuffle) = 1/20! ≈ 4e-19.
	ds := makeDataset(20)
	train1, _, err := dataset.Split(ds, 0.8, 42)
	if err != nil {
		t.Fatalf("seed 42: error inesperado: %v", err)
	}
	train2, _, err := dataset.Split(ds, 0.8, 99)
	if err != nil {
		t.Fatalf("seed 99: error inesperado: %v", err)
	}

	same := true
	for i := range train1.Points {
		if train1.Points[i] != train2.Points[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("seeds distintas produjeron el mismo split (estadísticamente improbable)")
	}
}

func TestSplit_SmallDataset(t *testing.T) {
	// Dataset mínimo (2 puntos): debe producir 1 train y 1 validation.
	ds := makeDataset(2)
	train, val, err := dataset.Split(ds, 0.8, 42)
	if err != nil {
		t.Fatalf("error inesperado: %v", err)
	}
	if train.Len() < 1 {
		t.Error("train debe tener al menos 1 punto")
	}
	if val.Len() < 1 {
		t.Error("val debe tener al menos 1 punto")
	}
	if train.Len()+val.Len() != 2 {
		t.Errorf("train(%d) + val(%d) != 2", train.Len(), val.Len())
	}
}

func TestSplit_NilDataset(t *testing.T) {
	_, _, err := dataset.Split(nil, 0.8, 42)
	if err == nil {
		t.Error("se esperaba error para dataset nil, se obtuvo nil")
	}
}

func TestSplit_EmptyDataset(t *testing.T) {
	_, _, err := dataset.Split(dataset.New(nil), 0.8, 42)
	if err == nil {
		t.Error("se esperaba error para dataset vacío, se obtuvo nil")
	}
}

func TestSplit_InvalidRatio(t *testing.T) {
	ds := makeDataset(10)
	for _, ratio := range []float64{0, -0.1, 1, 1.5} {
		_, _, err := dataset.Split(ds, ratio, 42)
		if err == nil {
			t.Errorf("ratio=%v: se esperaba error por ratio inválido, se obtuvo nil", ratio)
		}
	}
}

// ─── Generate ──────────────────────────────────────────────────────────────

func TestGenerate_Linear(t *testing.T) {
	// f(x) = 2*x + 1 en x = 1, 2, 3, 4, 5
	fn := func(x float64) float64 { return 2*x + 1 }
	ds := dataset.Generate(fn, 1, 5, 5)

	if ds.Len() != 5 {
		t.Fatalf("Len() = %d, want 5", ds.Len())
	}
	expected := []dataset.Point{{X: 1, Y: 3}, {X: 2, Y: 5}, {X: 3, Y: 7}, {X: 4, Y: 9}, {X: 5, Y: 11}}
	for i, p := range ds.Points {
		if p != expected[i] {
			t.Errorf("points[%d] = %v, want %v", i, p, expected[i])
		}
	}
}

func TestGenerate_Quadratic(t *testing.T) {
	// f(x) = x*x — función generadora de datos, no implica NodePow
	fn := func(x float64) float64 { return x * x }
	ds := dataset.Generate(fn, 1, 3, 3)

	if ds.Len() != 3 {
		t.Fatalf("Len() = %d, want 3", ds.Len())
	}
	for _, p := range ds.Points {
		want := p.X * p.X
		if p.Y != want {
			t.Errorf("punto {X:%v}: Y=%v, want %v", p.X, p.Y, want)
		}
	}
}

func TestGenerate_Linear2(t *testing.T) {
	// f(x) = 3*x - 2
	fn := func(x float64) float64 { return 3*x - 2 }
	ds := dataset.Generate(fn, 1, 4, 4)

	if ds.Len() != 4 {
		t.Fatalf("Len() = %d, want 4", ds.Len())
	}
	for _, p := range ds.Points {
		want := 3*p.X - 2
		if p.Y != want {
			t.Errorf("punto {X:%v}: Y=%v, want %v", p.X, p.Y, want)
		}
	}
}

func TestGenerate_LastPointExact(t *testing.T) {
	// El último punto debe ser exactamente xMax, no xMin + (n-1)*step.
	// Esto verifica que se usa el anclaje para evitar drift de float64.
	fn := func(x float64) float64 { return x }
	xMax := 1.0
	ds := dataset.Generate(fn, 0, xMax, 4) // step = 1/3, drift acumulado sin anclaje
	last := ds.Points[ds.Len()-1]
	if last.X != xMax {
		t.Errorf("último X = %v, want exactamente %v (drift de float64)", last.X, xMax)
	}
}

func TestGenerate_Deterministic(t *testing.T) {
	// Dos llamadas con los mismos parámetros deben producir resultados idénticos.
	fn := func(x float64) float64 { return x*x + 2*x - 1 }
	ds1 := dataset.Generate(fn, 0, 10, 11)
	ds2 := dataset.Generate(fn, 0, 10, 11)

	if ds1.Len() != ds2.Len() {
		t.Fatalf("tamaños distintos: %d vs %d", ds1.Len(), ds2.Len())
	}
	for i := range ds1.Points {
		if ds1.Points[i] != ds2.Points[i] {
			t.Errorf("points[%d]: %v != %v", i, ds1.Points[i], ds2.Points[i])
		}
	}
}

func TestGenerate_SinglePoint(t *testing.T) {
	fn := func(x float64) float64 { return x * 2 }
	ds := dataset.Generate(fn, 5, 5, 1)
	if ds.Len() != 1 {
		t.Fatalf("Len() = %d, want 1", ds.Len())
	}
	if ds.Points[0].X != 5 || ds.Points[0].Y != 10 {
		t.Errorf("points[0] = %v, want {5 10}", ds.Points[0])
	}
}

func TestGenerate_ZeroPoints(t *testing.T) {
	fn := func(x float64) float64 { return x }
	ds := dataset.Generate(fn, 0, 10, 0)
	if ds.Len() != 0 {
		t.Errorf("Len() = %d, want 0", ds.Len())
	}
}
