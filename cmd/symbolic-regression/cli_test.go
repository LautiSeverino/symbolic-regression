package main

import (
	"bytes"
	"errors"
	"flag"
	"io"
	"testing"

	"github.com/LautiSeverino/symbolic-regression/internal/config"
)

// ─── parseCLI — valores predeterminados ───────────────────────────────────

func TestParseCLI_Defaults(t *testing.T) {
	opts, err := parseCLI([]string{}, io.Discard)
	if err != nil {
		t.Fatalf("sin argumentos: error inesperado: %v", err)
	}
	if opts.DataPath != defaultDataPath {
		t.Errorf("DataPath = %q, want %q", opts.DataPath, defaultDataPath)
	}
	if opts.Profile != "balanced" {
		t.Errorf("Profile = %q, want %q", opts.Profile, "balanced")
	}
	if opts.Seed != 42 {
		t.Errorf("Seed = %d, want 42", opts.Seed)
	}
	if opts.Population != 0 {
		t.Errorf("Population = %d, want 0 (centinela)", opts.Population)
	}
	if opts.Generations != 0 {
		t.Errorf("Generations = %d, want 0 (centinela)", opts.Generations)
	}
	if opts.Workers != 0 {
		t.Errorf("Workers = %d, want 0 (centinela)", opts.Workers)
	}
	if opts.Depth != 0 {
		t.Errorf("Depth = %d, want 0 (centinela)", opts.Depth)
	}
	if opts.Nodes != 0 {
		t.Errorf("Nodes = %d, want 0 (centinela)", opts.Nodes)
	}
	if len(opts.Explicit) != 0 {
		t.Errorf("Explicit = %v, want vacío", opts.Explicit)
	}
}

// ─── parseCLI — parsing de los ocho flags ─────────────────────────────────

func TestParseCLI_AllEightFlags(t *testing.T) {
	args := []string{
		"--data", "datasets/quadratic.csv",
		"--profile", "fast",
		"--seed", "99",
		"--population", "500",
		"--generations", "100",
		"--workers", "4",
		"--depth", "6",
		"--nodes", "50",
	}
	opts, err := parseCLI(args, io.Discard)
	if err != nil {
		t.Fatalf("todos los flags válidos: error inesperado: %v", err)
	}
	if opts.DataPath != "datasets/quadratic.csv" {
		t.Errorf("DataPath = %q, want %q", opts.DataPath, "datasets/quadratic.csv")
	}
	if opts.Profile != "fast" {
		t.Errorf("Profile = %q, want %q", opts.Profile, "fast")
	}
	if opts.Seed != 99 {
		t.Errorf("Seed = %d, want 99", opts.Seed)
	}
	if opts.Population != 500 {
		t.Errorf("Population = %d, want 500", opts.Population)
	}
	if opts.Generations != 100 {
		t.Errorf("Generations = %d, want 100", opts.Generations)
	}
	if opts.Workers != 4 {
		t.Errorf("Workers = %d, want 4", opts.Workers)
	}
	if opts.Depth != 6 {
		t.Errorf("Depth = %d, want 6", opts.Depth)
	}
	if opts.Nodes != 50 {
		t.Errorf("Nodes = %d, want 50", opts.Nodes)
	}
}

// ─── parseCLI — detección de flags explícitos ─────────────────────────────

func TestParseCLI_ExplicitDetection_OnlySpecified(t *testing.T) {
	opts, err := parseCLI([]string{"--seed", "7", "--population", "300"}, io.Discard)
	if err != nil {
		t.Fatalf("error inesperado: %v", err)
	}
	if !opts.Explicit["seed"] {
		t.Error("seed debe ser explícito")
	}
	if !opts.Explicit["population"] {
		t.Error("population debe ser explícito")
	}
	for _, name := range []string{"data", "profile", "generations", "workers", "depth", "nodes"} {
		if opts.Explicit[name] {
			t.Errorf("flag %q no debe ser explícito", name)
		}
	}
}

func TestParseCLI_ExplicitDetection_AllEight(t *testing.T) {
	args := []string{
		"--data", "x.csv", "--profile", "fast", "--seed", "1",
		"--population", "10", "--generations", "5",
		"--workers", "1", "--depth", "3", "--nodes", "10",
	}
	opts, err := parseCLI(args, io.Discard)
	if err != nil {
		t.Fatalf("error inesperado: %v", err)
	}
	for _, name := range []string{"data", "profile", "seed", "population", "generations", "workers", "depth", "nodes"} {
		if !opts.Explicit[name] {
			t.Errorf("flag %q debe ser explícito", name)
		}
	}
}

func TestParseCLI_ExplicitDetection_NoneWhenNoArgs(t *testing.T) {
	opts, err := parseCLI([]string{}, io.Discard)
	if err != nil {
		t.Fatalf("error inesperado: %v", err)
	}
	if len(opts.Explicit) != 0 {
		t.Errorf("sin flags: Explicit debe estar vacío, got %v", opts.Explicit)
	}
}

// ─── parseCLI — argumento posicional (compatibilidad hacia atrás) ─────────

func TestParseCLI_PositionalArg_UsedWhenNoDataFlag(t *testing.T) {
	opts, err := parseCLI([]string{"datasets/quadratic.csv"}, io.Discard)
	if err != nil {
		t.Fatalf("argumento posicional: error inesperado: %v", err)
	}
	if opts.DataPath != "datasets/quadratic.csv" {
		t.Errorf("DataPath = %q, want %q", opts.DataPath, "datasets/quadratic.csv")
	}
	if opts.Explicit["data"] {
		t.Error("argumento posicional no debe marcar 'data' como explícito")
	}
}

func TestParseCLI_PositionalArg_WithOtherFlags(t *testing.T) {
	args := []string{"--seed", "7", "--profile", "fast", "datasets/linear.csv"}
	opts, err := parseCLI(args, io.Discard)
	if err != nil {
		t.Fatalf("error inesperado: %v", err)
	}
	if opts.DataPath != "datasets/linear.csv" {
		t.Errorf("DataPath = %q, want %q", opts.DataPath, "datasets/linear.csv")
	}
	if opts.Seed != 7 {
		t.Errorf("Seed = %d, want 7", opts.Seed)
	}
}

// ─── parseCLI — precedencia de --data ────────────────────────────────────

func TestParseCLI_DataFlag_TakesPrecedenceOverDefault(t *testing.T) {
	opts, err := parseCLI([]string{"--data", "mydata.csv"}, io.Discard)
	if err != nil {
		t.Fatalf("error inesperado: %v", err)
	}
	if opts.DataPath != "mydata.csv" {
		t.Errorf("DataPath = %q, want %q", opts.DataPath, "mydata.csv")
	}
	if !opts.Explicit["data"] {
		t.Error("'data' debe ser explícito cuando se usa --data")
	}
}

// ─── parseCLI — errores de ambigüedad y posicionales ─────────────────────

func TestParseCLI_DataAndPositional_ReturnsError(t *testing.T) {
	_, err := parseCLI([]string{"--data", "a.csv", "b.csv"}, io.Discard)
	if err == nil {
		t.Error("--data + posicional: se esperaba error por ambigüedad, se obtuvo nil")
	}
	if errors.Is(err, flag.ErrHelp) {
		t.Error("error de ambigüedad no debe ser ErrHelp")
	}
}

func TestParseCLI_MultiplePositional_ReturnsError(t *testing.T) {
	_, err := parseCLI([]string{"a.csv", "b.csv"}, io.Discard)
	if err == nil {
		t.Error("múltiples posicionales: se esperaba error, se obtuvo nil")
	}
}

// ─── parseCLI — flags desconocidos y valores inválidos ────────────────────

func TestParseCLI_UnknownFlag_ReturnsError(t *testing.T) {
	_, err := parseCLI([]string{"--unknownflag"}, io.Discard)
	if err == nil {
		t.Error("flag desconocido: se esperaba error, se obtuvo nil")
	}
	if errors.Is(err, flag.ErrHelp) {
		t.Error("flag desconocido: error no debe ser ErrHelp")
	}
}

func TestParseCLI_InvalidIntFlag_ReturnsError(t *testing.T) {
	cases := []struct {
		name string
		args []string
	}{
		{"population no entero", []string{"--population", "abc"}},
		{"generations no entero", []string{"--generations", "1.5"}},
		{"workers no entero", []string{"--workers", "dos"}},
		{"depth no entero", []string{"--depth", "siete"}},
		{"nodes no entero", []string{"--nodes", "?"}},
		{"seed no entero", []string{"--seed", "nada"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parseCLI(tc.args, io.Discard)
			if err == nil {
				t.Errorf("%s: se esperaba error por valor no entero, se obtuvo nil", tc.name)
			}
		})
	}
}

// ─── parseCLI — ayuda ─────────────────────────────────────────────────────

func TestParseCLI_Help(t *testing.T) {
	for _, arg := range []string{"--help", "-h"} {
		var buf bytes.Buffer
		_, err := parseCLI([]string{arg}, &buf)
		if !errors.Is(err, flag.ErrHelp) {
			t.Errorf("%s: got %v, want flag.ErrHelp", arg, err)
		}
		if buf.Len() == 0 {
			t.Errorf("%s: no se escribió salida de ayuda", arg)
		}
	}
}

// ─── applyConfigOverrides — ningún override conserva el perfil ────────────

func TestApplyConfigOverrides_NoExplicit_PreservesAllProfileValues(t *testing.T) {
	base := config.BALANCEDConfig()
	base.Seed = 42
	base.GenConfig.Seed = 43

	opts := cliOptions{Explicit: map[string]bool{}}
	got := applyConfigOverrides(base, opts)

	if got.PopulationSize != base.PopulationSize {
		t.Errorf("PopulationSize: got %d, want %d", got.PopulationSize, base.PopulationSize)
	}
	if got.Generations != base.Generations {
		t.Errorf("Generations: got %d, want %d", got.Generations, base.Generations)
	}
	if got.Workers != base.Workers {
		t.Errorf("Workers: got %d, want %d", got.Workers, base.Workers)
	}
	if got.GenConfig.MaxDepth != base.GenConfig.MaxDepth {
		t.Errorf("MaxDepth: got %d, want %d", got.GenConfig.MaxDepth, base.GenConfig.MaxDepth)
	}
	if got.GenConfig.MaxNodes != base.GenConfig.MaxNodes {
		t.Errorf("MaxNodes: got %d, want %d", got.GenConfig.MaxNodes, base.GenConfig.MaxNodes)
	}
}

// ─── applyConfigOverrides — overrides individuales ────────────────────────

func TestApplyConfigOverrides_PopulationOnly(t *testing.T) {
	base := config.BALANCEDConfig()
	base.Seed = 42
	base.GenConfig.Seed = 43

	opts := cliOptions{Population: 2000, Explicit: map[string]bool{"population": true}}
	got := applyConfigOverrides(base, opts)

	if got.PopulationSize != 2000 {
		t.Errorf("PopulationSize = %d, want 2000", got.PopulationSize)
	}
	if got.Generations != base.Generations {
		t.Errorf("Generations cambió: %d → %d", base.Generations, got.Generations)
	}
	if got.Workers != base.Workers {
		t.Errorf("Workers cambió: %d → %d", base.Workers, got.Workers)
	}
}

func TestApplyConfigOverrides_GenerationsOnly(t *testing.T) {
	base := config.FASTConfig()
	base.Seed = 1
	base.GenConfig.Seed = 2

	opts := cliOptions{Generations: 200, Explicit: map[string]bool{"generations": true}}
	got := applyConfigOverrides(base, opts)

	if got.Generations != 200 {
		t.Errorf("Generations = %d, want 200", got.Generations)
	}
	if got.PopulationSize != base.PopulationSize {
		t.Errorf("PopulationSize cambió inesperadamente: %d → %d", base.PopulationSize, got.PopulationSize)
	}
}

func TestApplyConfigOverrides_WorkersOnly(t *testing.T) {
	base := config.BALANCEDConfig()
	base.Seed = 42
	base.GenConfig.Seed = 43

	opts := cliOptions{Workers: 2, Explicit: map[string]bool{"workers": true}}
	got := applyConfigOverrides(base, opts)

	if got.Workers != 2 {
		t.Errorf("Workers = %d, want 2", got.Workers)
	}
	if got.PopulationSize != base.PopulationSize {
		t.Errorf("PopulationSize cambió inesperadamente")
	}
}

func TestApplyConfigOverrides_DepthOnly(t *testing.T) {
	base := config.BALANCEDConfig()
	base.Seed = 42
	base.GenConfig.Seed = 43
	originalNodes := base.GenConfig.MaxNodes

	opts := cliOptions{Depth: 10, Explicit: map[string]bool{"depth": true}}
	got := applyConfigOverrides(base, opts)

	if got.GenConfig.MaxDepth != 10 {
		t.Errorf("MaxDepth = %d, want 10", got.GenConfig.MaxDepth)
	}
	if got.GenConfig.MaxNodes != originalNodes {
		t.Errorf("MaxNodes cambió de %d a %d (no debe cambiar)", originalNodes, got.GenConfig.MaxNodes)
	}
}

func TestApplyConfigOverrides_NodesOnly(t *testing.T) {
	base := config.BALANCEDConfig()
	base.Seed = 42
	base.GenConfig.Seed = 43
	originalDepth := base.GenConfig.MaxDepth

	opts := cliOptions{Nodes: 50, Explicit: map[string]bool{"nodes": true}}
	got := applyConfigOverrides(base, opts)

	if got.GenConfig.MaxNodes != 50 {
		t.Errorf("MaxNodes = %d, want 50", got.GenConfig.MaxNodes)
	}
	if got.GenConfig.MaxDepth != originalDepth {
		t.Errorf("MaxDepth cambió de %d a %d (no debe cambiar)", originalDepth, got.GenConfig.MaxDepth)
	}
}

// ─── applyConfigOverrides — depth y nodes son independientes ──────────────

func TestApplyConfigOverrides_DepthAndNodes_Independent(t *testing.T) {
	base := config.BALANCEDConfig()
	base.Seed = 42
	base.GenConfig.Seed = 43

	opts := cliOptions{
		Depth: 10, Nodes: 100,
		Explicit: map[string]bool{"depth": true, "nodes": true},
	}
	got := applyConfigOverrides(base, opts)

	if got.GenConfig.MaxDepth != 10 {
		t.Errorf("MaxDepth = %d, want 10", got.GenConfig.MaxDepth)
	}
	if got.GenConfig.MaxNodes != 100 {
		t.Errorf("MaxNodes = %d, want 100", got.GenConfig.MaxNodes)
	}
}

func TestApplyConfigOverrides_Depth_DoesNotAffectGenConfigOtherFields(t *testing.T) {
	base := config.BALANCEDConfig()
	base.Seed = 42
	base.GenConfig.Seed = 43
	originalBinaryOps := len(base.GenConfig.AllowedBinaryOps)
	originalUnaryOps := len(base.GenConfig.AllowedUnaryOps)
	originalConstMin := base.GenConfig.ConstantMin
	originalConstMax := base.GenConfig.ConstantMax

	opts := cliOptions{Depth: 9, Explicit: map[string]bool{"depth": true}}
	got := applyConfigOverrides(base, opts)

	if len(got.GenConfig.AllowedBinaryOps) != originalBinaryOps {
		t.Error("AllowedBinaryOps cambió (no debe cambiar)")
	}
	if len(got.GenConfig.AllowedUnaryOps) != originalUnaryOps {
		t.Error("AllowedUnaryOps cambió (no debe cambiar)")
	}
	if got.GenConfig.ConstantMin != originalConstMin {
		t.Error("ConstantMin cambió (no debe cambiar)")
	}
	if got.GenConfig.ConstantMax != originalConstMax {
		t.Error("ConstantMax cambió (no debe cambiar)")
	}
}

// ─── applyConfigOverrides — todos los overrides simultáneos ──────────────

func TestApplyConfigOverrides_AllFiveOverrides(t *testing.T) {
	base := config.FASTConfig()
	base.Seed = 1
	base.GenConfig.Seed = 2

	opts := cliOptions{
		Population:  500,
		Generations: 100,
		Workers:     2,
		Depth:       6,
		Nodes:       60,
		Explicit: map[string]bool{
			"population":  true,
			"generations": true,
			"workers":     true,
			"depth":       true,
			"nodes":       true,
		},
	}
	got := applyConfigOverrides(base, opts)

	if got.PopulationSize != 500 {
		t.Errorf("PopulationSize = %d, want 500", got.PopulationSize)
	}
	if got.Generations != 100 {
		t.Errorf("Generations = %d, want 100", got.Generations)
	}
	if got.Workers != 2 {
		t.Errorf("Workers = %d, want 2", got.Workers)
	}
	if got.GenConfig.MaxDepth != 6 {
		t.Errorf("MaxDepth = %d, want 6", got.GenConfig.MaxDepth)
	}
	if got.GenConfig.MaxNodes != 60 {
		t.Errorf("MaxNodes = %d, want 60", got.GenConfig.MaxNodes)
	}
}

// ─── applyConfigOverrides — override explícito de cero se aplica ─────────

func TestApplyConfigOverrides_ExplicitZeroIsApplied(t *testing.T) {
	// Si el usuario pasa --population 0 explícitamente, el override se aplica y
	// cfg.PopulationSize queda en 0. La validación (must be ≥ 2) ocurre
	// posteriormente en genetic.NewEvolution, no en applyConfigOverrides.
	base := config.BALANCEDConfig()
	base.Seed = 42
	base.GenConfig.Seed = 43

	cases := []struct {
		name    string
		opts    cliOptions
		wantPop int
		wantGen int
	}{
		{
			"population=0 explícito",
			cliOptions{Population: 0, Explicit: map[string]bool{"population": true}},
			0, base.Generations,
		},
		{
			"generations=0 explícito",
			cliOptions{Generations: 0, Explicit: map[string]bool{"generations": true}},
			base.PopulationSize, 0,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := applyConfigOverrides(base, tc.opts)
			if got.PopulationSize != tc.wantPop {
				t.Errorf("PopulationSize = %d, want %d", got.PopulationSize, tc.wantPop)
			}
			if got.Generations != tc.wantGen {
				t.Errorf("Generations = %d, want %d", got.Generations, tc.wantGen)
			}
		})
	}
}

// ─── applyConfigOverrides — el override gana sobre cualquier perfil ───────

func TestApplyConfigOverrides_OverrideWinsOverProfile(t *testing.T) {
	// INTENSIVE tiene PopulationSize=5000.
	// Con --population 50, el resultado debe ser 50.
	base := config.INTENSIVEConfig()
	base.Seed = 1
	base.GenConfig.Seed = 2

	opts := cliOptions{Population: 50, Explicit: map[string]bool{"population": true}}
	got := applyConfigOverrides(base, opts)

	if got.PopulationSize != 50 {
		t.Errorf("PopulationSize = %d, want 50 (override sobre INTENSIVE)", got.PopulationSize)
	}
}

func TestApplyConfigOverrides_ProfileValuesPreservedForNonExplicitFlags(t *testing.T) {
	// INTENSIVE: pop=5000, gen=1000, depth=10, nodes=100, workers=8.
	// Solo --population=50 es explícito; el resto debe conservar INTENSIVE.
	base := config.INTENSIVEConfig()
	base.Seed = 1
	base.GenConfig.Seed = 2

	opts := cliOptions{Population: 50, Explicit: map[string]bool{"population": true}}
	got := applyConfigOverrides(base, opts)

	if got.Generations != 1000 {
		t.Errorf("Generations = %d, want 1000 (perfil INTENSIVE)", got.Generations)
	}
	if got.Workers != 8 {
		t.Errorf("Workers = %d, want 8 (perfil INTENSIVE)", got.Workers)
	}
	if got.GenConfig.MaxDepth != 10 {
		t.Errorf("MaxDepth = %d, want 10 (perfil INTENSIVE)", got.GenConfig.MaxDepth)
	}
	if got.GenConfig.MaxNodes != 100 {
		t.Errorf("MaxNodes = %d, want 100 (perfil INTENSIVE)", got.GenConfig.MaxNodes)
	}
}

// ─── applyConfigOverrides — Workers=0 explícito se aplica ────────────────

func TestApplyConfigOverrides_WorkersZeroExplicit_IsApplied(t *testing.T) {
	// --workers 0 explícito: cfg.Workers queda en 0 tras applyConfigOverrides.
	// main() llama a normalizeConfig() que lo normaliza a 1 antes de mostrar
	// la configuración y de llamar a genetic.NewEvolution (semántica de Fase 10).
	base := config.BALANCEDConfig()
	base.Seed = 42
	base.GenConfig.Seed = 43

	opts := cliOptions{Workers: 0, Explicit: map[string]bool{"workers": true}}
	got := applyConfigOverrides(base, opts)

	if got.Workers != 0 {
		t.Errorf("Workers = %d, want 0 (normalización ocurre en NewEvolution, no aquí)", got.Workers)
	}
}

// ─── parseCLI — default de --data muestra el dataset predeterminado ───────

func TestParseCLI_DataFlagDefault_IsDefaultDataPath(t *testing.T) {
	// Sin pasar ningún argumento, --data debe mostrar el default en la ayuda con
	// el formato exacto que produce flag.PrintDefaults: (default "datasets/linear.csv").
	// Esto verifica que fs.String("data", defaultDataPath, ...) está declarado
	// correctamente y no con default vacío.
	var buf bytes.Buffer
	_, err := parseCLI([]string{"--help"}, &buf)
	if !errors.Is(err, flag.ErrHelp) {
		t.Fatalf("--help: got %v, want flag.ErrHelp", err)
	}
	wantSubstr := `(default "` + defaultDataPath + `")`
	helpText := buf.String()
	if !contains(helpText, wantSubstr) {
		t.Errorf("la ayuda no contiene %q\ngot:\n%s", wantSubstr, helpText)
	}
}

// contains es un helper local para no importar strings en el nivel del package.
func contains(s, substr string) bool {
	return len(s) >= len(substr) && func() bool {
		for i := 0; i <= len(s)-len(substr); i++ {
			if s[i:i+len(substr)] == substr {
				return true
			}
		}
		return false
	}()
}

// ─── normalizeConfig ──────────────────────────────────────────────────────

func TestNormalizeConfig_WorkersZeroBecomesOne(t *testing.T) {
	base := config.BALANCEDConfig()
	base.Workers = 0
	got := normalizeConfig(base)
	if got.Workers != 1 {
		t.Errorf("Workers = %d, want 1 (Workers=0 debe normalizarse a 1)", got.Workers)
	}
}

func TestNormalizeConfig_WorkersNegativeBecomesOne(t *testing.T) {
	base := config.BALANCEDConfig()
	base.Workers = -5
	got := normalizeConfig(base)
	if got.Workers != 1 {
		t.Errorf("Workers = %d, want 1 (Workers<0 debe normalizarse a 1)", got.Workers)
	}
}

func TestNormalizeConfig_WorkersOneUnchanged(t *testing.T) {
	base := config.BALANCEDConfig()
	base.Workers = 1
	got := normalizeConfig(base)
	if got.Workers != 1 {
		t.Errorf("Workers = %d, want 1 (Workers=1 no debe cambiar)", got.Workers)
	}
}

func TestNormalizeConfig_WorkersPositiveUnchanged(t *testing.T) {
	base := config.BALANCEDConfig()
	base.Workers = 8
	got := normalizeConfig(base)
	if got.Workers != 8 {
		t.Errorf("Workers = %d, want 8 (Workers>0 no debe cambiar)", got.Workers)
	}
}

func TestNormalizeConfig_DoesNotAffectOtherFields(t *testing.T) {
	// normalizeConfig solo toca Workers; el resto del Config no debe cambiar.
	base := config.BALANCEDConfig()
	base.Workers = 0
	got := normalizeConfig(base)

	if got.PopulationSize != base.PopulationSize {
		t.Errorf("PopulationSize cambió: %d → %d", base.PopulationSize, got.PopulationSize)
	}
	if got.Generations != base.Generations {
		t.Errorf("Generations cambió: %d → %d", base.Generations, got.Generations)
	}
	if got.GenConfig.MaxDepth != base.GenConfig.MaxDepth {
		t.Errorf("MaxDepth cambió")
	}
	if got.GenConfig.MaxNodes != base.GenConfig.MaxNodes {
		t.Errorf("MaxNodes cambió")
	}
}
