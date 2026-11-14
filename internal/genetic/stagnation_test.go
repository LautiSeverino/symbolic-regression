package genetic

// stagnation_test.go — Fase 14: tests white-box de updateStagnationCount
// y de la propiedad de clonación de GenerationsRun en cloneIndividual.
//
// Este archivo usa package genetic (no package genetic_test) para acceder
// a las funciones internas updateStagnationCount y cloneIndividual.

import (
	"testing"

	"github.com/LautiSeverino/symbolic-regression/internal/expression"
	"github.com/LautiSeverino/symbolic-regression/internal/fitness"
)

// ─── updateStagnationCount ────────────────────────────────────────────────

func TestStagnationCount_NoImprovement_Increments(t *testing.T) {
	// improvement = 5.0 - 5.0 = 0; 0 > 0 es false → incrementa.
	got := updateStagnationCount(5.0, 5.0, 0, 0)
	if got != 1 {
		t.Errorf("got %d, want 1", got)
	}
}

func TestStagnationCount_NoImprovement_AccumulatesCount(t *testing.T) {
	// El contador acumula desde count=3.
	got := updateStagnationCount(5.0, 5.0, 0, 3)
	if got != 4 {
		t.Errorf("got %d, want 4", got)
	}
}

func TestStagnationCount_StrictImprovement_Resets(t *testing.T) {
	// improvement = 5.0 - 3.0 = 2.0; 2.0 > 0 es true → resetea.
	got := updateStagnationCount(5.0, 3.0, 0, 7)
	if got != 0 {
		t.Errorf("got %d, want 0", got)
	}
}

func TestStagnationCount_ZeroDelta_AnyStrictImprovementResets(t *testing.T) {
	// Con delta=0 cualquier mejora estricta resetea.
	// 5.0 - 4.875 = 0.125; exacto en binario; 0.125 > 0 es true.
	got := updateStagnationCount(5.0, 4.875, 0, 10)
	if got != 0 {
		t.Errorf("got %d, want 0", got)
	}
}

func TestStagnationCount_ImprovementBelowDelta_Increments(t *testing.T) {
	// improvement = 5.0 - 4.9375 = 0.0625; delta = 0.125; 0.0625 > 0.125 es false.
	// Todos los valores son exactos en binario (potencias de 2).
	got := updateStagnationCount(5.0, 4.9375, 0.125, 2)
	if got != 3 {
		t.Errorf("got %d, want 3 (mejora 0.0625 < delta 0.125 → incrementa)", got)
	}
}

func TestStagnationCount_ImprovementAboveDelta_Resets(t *testing.T) {
	// improvement = 5.0 - 4.75 = 0.25; delta = 0.125; 0.25 > 0.125 es true.
	got := updateStagnationCount(5.0, 4.75, 0.125, 5)
	if got != 0 {
		t.Errorf("got %d, want 0 (mejora 0.25 > delta 0.125 → resetea)", got)
	}
}

func TestStagnationCount_ExactDelta_DoesNotReset(t *testing.T) {
	// improvement = 5.0 - 4.875 = 0.125; delta = 0.125; 0.125 > 0.125 es false.
	// La condición es estrictamente mayor (>); igualdad exacta no resetea.
	// Todos los valores son exactos en binario: 5.0, 4.875 = 4 + 7/8, 0.125 = 1/8.
	got := updateStagnationCount(5.0, 4.875, 0.125, 3)
	if got != 4 {
		t.Errorf("got %d, want 4 (improvement == delta → NO resetea; condición es >)", got)
	}
}

func TestStagnationCount_FromZero_NoImprovement(t *testing.T) {
	// Caso límite: fitness en 0, sin posibilidad de mejorar.
	got := updateStagnationCount(0, 0, 0, 0)
	if got != 1 {
		t.Errorf("got %d, want 1", got)
	}
}

// TestStagnationCount_MultipleSmallImprovements_BelowDelta_AllIncrement documenta
// que la comparación es estrictamente generación a generación, no acumulada.
// Varias mejoras consecutivas, cada una inferior a delta, incrementan el contador
// aunque su suma supere delta.
//
// Con delta=0.125 y improvement=0.0625 por generación (ambos exactos en binario):
//
//	gen 1: prev=5.0000, next=4.9375 → improvement=0.0625 < 0.125 → count=1
//	gen 2: prev=4.9375, next=4.8750 → improvement=0.0625 < 0.125 → count=2
//	gen 3: prev=4.8750, next=4.8125 → improvement=0.0625 < 0.125 → count=3
//
// La mejora acumulada es 0.1875 > 0.125, pero el contador llega a 3.
func TestStagnationCount_MultipleSmallImprovements_BelowDelta_AllIncrement(t *testing.T) {
	const delta = 0.125 // 1/8, exacto en binario

	type step struct {
		prev, next float64
		wantCount  int
	}
	steps := []step{
		{5.0, 4.9375, 1},   // improvement = 0.0625 (1/16), < delta
		{4.9375, 4.875, 2}, // improvement = 0.0625 (1/16), < delta
		{4.875, 4.8125, 3}, // improvement = 0.0625 (1/16), < delta
	}

	count := 0
	for i, s := range steps {
		count = updateStagnationCount(s.prev, s.next, delta, count)
		if count != s.wantCount {
			t.Errorf(
				"paso %d: got count=%d, want %d "+
					"(la mejora acumulada %.4f no previene el incremento cuando "+
					"cada mejora individual %.4f < delta %.3f)",
				i+1, count, s.wantCount,
				5.0-s.next, s.prev-s.next, delta,
			)
		}
	}
}

// ─── cloneIndividual — GenerationsRun ─────────────────────────────────────

func TestCloneIndividual_PreservesGenerationsRun(t *testing.T) {
	ind := Individual{
		Expr:           expression.NewVariable(),
		Result:         fitness.FitnessResult{MSE: 1.0, Fitness: 1.01, Complexity: 1},
		GenerationsRun: 42,
	}
	clone := cloneIndividual(ind)
	if clone.GenerationsRun != 42 {
		t.Errorf("cloneIndividual: GenerationsRun = %d, want 42", clone.GenerationsRun)
	}
}

func TestCloneIndividual_GenerationsRun_Zero(t *testing.T) {
	// GenerationsRun = 0 (zero value) también debe copiarse correctamente.
	ind := Individual{
		Expr:   expression.NewVariable(),
		Result: fitness.FitnessResult{},
	}
	clone := cloneIndividual(ind)
	if clone.GenerationsRun != 0 {
		t.Errorf("cloneIndividual: GenerationsRun = %d, want 0", clone.GenerationsRun)
	}
}

func TestCloneIndividual_GenerationsRun_Independence(t *testing.T) {
	// GenerationsRun es int (tipo valor). Modificar el original no afecta al clon.
	ind := Individual{
		Expr:           expression.NewVariable(),
		Result:         fitness.FitnessResult{},
		GenerationsRun: 10,
	}
	clone := cloneIndividual(ind)
	ind.GenerationsRun = 999
	if clone.GenerationsRun != 10 {
		t.Errorf("cloneIndividual: modificar original afectó al clon: got %d, want 10",
			clone.GenerationsRun)
	}
}
