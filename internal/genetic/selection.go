package genetic

import "sort"

// sortByFitness ordena pop de menor a mayor Fitness (menor es mejor).
func sortByFitness(pop []Individual) {
	sort.Slice(pop, func(i, j int) bool {
		return pop[i].Result.Fitness < pop[j].Result.Fitness
	})
}

// TournamentSelect elige al mejor entre TournamentSize candidatos aleatorios
// seleccionados con reemplazo de pop.
//
// El individuo retornado comparte su Expr con el elemento de pop —
// clonar antes de modificar o insertar en la nueva generación.
func (e *Evolution) TournamentSelect(pop []Individual) Individual {
	if len(pop) == 0 {
		panic("genetic: TournamentSelect called with empty population")
	}
	size := e.cfg.TournamentSize
	if size > len(pop) {
		size = len(pop)
	}
	best := pop[e.rng.Intn(len(pop))]
	for i := 1; i < size; i++ {
		candidate := pop[e.rng.Intn(len(pop))]
		if candidate.Result.Fitness < best.Result.Fitness {
			best = candidate
		}
	}
	return best
}
