package expression

import (
	"fmt"
	"math"
)

// EvalError es retornado cuando una expresión no puede evaluarse en un punto dado.
//
// Causas posibles:
//   - Constante no finita (NaN o ±Inf) — entrada inválida en nodo hoja
//   - División por cero
//   - Overflow aritmético (resultado NaN/Inf)
//   - sqrt de argumento negativo
//   - ln de argumento no positivo
//   - 0 elevado a exponente negativo
//   - Base negativa con exponente no entero (resultado complejo/NaN)
//   - exp con argumento que produce +Inf (overflow)
//
// Políticas numéricas adoptadas:
//   - 0^0 = 1    (convención Go/IEEE-754; math.Pow(0,0) = 1)
//   - exp(underflow) = 0   (desbordamiento a cero es un float64 válido, no error)
//   - abs, sin, cos de argumento finito → resultado siempre finito, no error
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
		// Guard: una constante no finita propagaría NaN/Inf silenciosamente por todo el árbol.
		// El generador y las mutaciones siempre producen constantes finitas, pero quien
		// construya nodos manualmente (tests, extensiones) podría producir valores inválidos.
		if math.IsNaN(node.Value) || math.IsInf(node.Value, 0) {
			return 0, &EvalError{Msg: fmt.Sprintf("non-finite constant: %v", node.Value)}
		}
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

	case NodePow:
		// Ambos hijos se evalúan como cualquier operador binario.
		l, r, err := evalBoth(node, x)
		if err != nil {
			return 0, err
		}
		// Caso especial explícito: 0 con exponente negativo → Inf.
		// finiteOr lo capturaría de todas formas, pero el mensaje es más claro así.
		if l == 0 && r < 0 {
			return 0, &EvalError{Msg: fmt.Sprintf("0 raised to negative exponent: %v", r)}
		}
		result := math.Pow(l, r)
		// finiteOr captura:
		//   - NaN  → base negativa con exponente no entero (e.g. (-2)^0.5)
		//   - ±Inf → overflow o Pow(0, neg) si no fue capturado arriba
		return finiteOr(result, "^", l, r)

	case NodeAbs:
		arg, err := Evaluate(node.Left, x)
		if err != nil {
			return 0, err
		}
		result := math.Abs(arg)
		// math.Abs es seguro para arg finito: siempre devuelve un valor en [0, +MaxFloat64].
		// El check es defensa en profundidad: con el guard en NodeConstant y la validación
		// del dataset, arg nunca debería ser no-finito en condiciones normales.
		if math.IsNaN(result) || math.IsInf(result, 0) {
			return 0, &EvalError{Msg: fmt.Sprintf("non-finite abs(%v)", arg)}
		}
		return result, nil

	case NodeExp:
		arg, err := Evaluate(node.Left, x)
		if err != nil {
			return 0, err
		}
		result := math.Exp(arg)
		// math.Exp puede producir +Inf (overflow) pero nunca NaN para arg finito.
		if math.IsInf(result, 0) {
			return 0, &EvalError{Msg: fmt.Sprintf("exp overflow: exp(%v)", arg)}
		}
		return result, nil

	case NodeSin:
		arg, err := Evaluate(node.Left, x)
		if err != nil {
			return 0, err
		}
		result := math.Sin(arg)
		// Para arg finito, sin siempre devuelve un valor en [-1, 1].
		// El chequeo es defensivo: cubriría el caso (teóricamente imposible aquí)
		// de que un arg finito produjera NaN/Inf.
		if math.IsNaN(result) || math.IsInf(result, 0) {
			return 0, &EvalError{Msg: fmt.Sprintf("non-finite sin(%v)", arg)}
		}
		return result, nil

	case NodeCos:
		arg, err := Evaluate(node.Left, x)
		if err != nil {
			return 0, err
		}
		result := math.Cos(arg)
		if math.IsNaN(result) || math.IsInf(result, 0) {
			return 0, &EvalError{Msg: fmt.Sprintf("non-finite cos(%v)", arg)}
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
