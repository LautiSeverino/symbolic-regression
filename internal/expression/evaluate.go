package expression

import (
	"fmt"
	"math"
)

// EvalError es retornado cuando una expresión no puede evaluarse en un punto dado.
// Causas: división por cero, overflow, NaN, sqrt de negativos, ln de no positivos.
// Fase 8 agregará: pow con exponentes inválidos, dominios de otras funciones.
type EvalError struct {
	Msg string
}

func (e *EvalError) Error() string { return "eval: " + e.Msg }

// Evaluate computa el valor de la expresión en x.
// Retorna EvalError en lugar de crashear ante operaciones inválidas.
// El caller (fitness) es responsable de convertir el error en penalización.
func Evaluate(node *Node, x float64) (float64, error) {
	if node == nil {
		return 0, &EvalError{Msg: "nil node"}
	}

	switch node.Type {

	case NodeVariable:
		return x, nil

	case NodeConstant:
		return node.Value, nil

	case NodeAdd:
		l, r, err := evalBoth(node, x)
		if err != nil {
			return 0, err
		}
		return finiteOr(l+r, "+", l, r)

	case NodeSub:
		l, r, err := evalBoth(node, x)
		if err != nil {
			return 0, err
		}
		return finiteOr(l-r, "-", l, r)

	case NodeMul:
		l, r, err := evalBoth(node, x)
		if err != nil {
			return 0, err
		}
		return finiteOr(l*r, "*", l, r)

	case NodeDiv:
		l, r, err := evalBoth(node, x)
		if err != nil {
			return 0, err
		}
		if r == 0 {
			return 0, &EvalError{Msg: "division by zero"}
		}
		return finiteOr(l/r, "/", l, r)

	case NodeSqrt:
		arg, err := Evaluate(node.Left, x)
		if err != nil {
			return 0, err
		}
		if arg < 0 {
			return 0, &EvalError{Msg: fmt.Sprintf("sqrt of negative: %v", arg)}
		}
		result := math.Sqrt(arg)
		if math.IsNaN(result) || math.IsInf(result, 0) {
			return 0, &EvalError{Msg: fmt.Sprintf("non-finite sqrt(%v)", arg)}
		}
		return result, nil

	case NodeLn:
		arg, err := Evaluate(node.Left, x)
		if err != nil {
			return 0, err
		}
		if arg <= 0 {
			return 0, &EvalError{Msg: fmt.Sprintf("ln of non-positive: %v", arg)}
		}
		result := math.Log(arg)
		if math.IsNaN(result) || math.IsInf(result, 0) {
			return 0, &EvalError{Msg: fmt.Sprintf("non-finite ln(%v)", arg)}
		}
		return result, nil

	default:
		return 0, &EvalError{Msg: fmt.Sprintf("unknown node type: %d", node.Type)}
	}
}

// evalBoth evalúa ambos hijos de un nodo binario.
func evalBoth(node *Node, x float64) (float64, float64, error) {
	l, err := Evaluate(node.Left, x)
	if err != nil {
		return 0, 0, err
	}
	r, err := Evaluate(node.Right, x)
	if err != nil {
		return 0, 0, err
	}
	return l, r, nil
}

// finiteOr devuelve result si es finito, o EvalError en caso de NaN/Inf.
// Detecta overflow en multiplicación, suma extrema, etc.
func finiteOr(result float64, op string, l, r float64) (float64, error) {
	if math.IsNaN(result) || math.IsInf(result, 0) {
		return 0, &EvalError{
			Msg: fmt.Sprintf("non-finite result: %v %s %v", l, op, r),
		}
	}
	return result, nil
}
