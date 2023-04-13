package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
)

var (
	// 入力から隠れ層への重み
	w_x = mat.NewDense(2, 3, []float64{0.5, 0.3, 0.7, 0.1, 0.3, 0.8})

	// 隠れ層から隠れ層への重み
	w_h = mat.NewDense(2, 2, []float64{0.6, 0.4, 0.2, 0.9})

	// 隠れ層のバイアス
	b_h = mat.NewDense(2, 1, []float64{0.1, 0.1})

	// 隠れ層から出力層への重み
	w_y = mat.NewDense(3, 2, []float64{1.0, 0.5, 0.3, 0.5, 0.7, 0.9})

	// 出力層のバイアス
	b_y = mat.NewDense(3, 1, []float64{0.1, 0.1, 0.1})

	// 入力シーケンス
	a = mat.NewDense(3, 1, []float64{1, 0, 0})
	b = mat.NewDense(3, 1, []float64{0, 1, 0})
	c = mat.NewDense(3, 1, []float64{0, 0, 1})
)

func main() {
	// 初期の隠れ状態
	h := mat.NewDense(2, 1, []float64{0, 0})

	// 入力を、{a, b, c, c, b, a}とする
	inputs := []*mat.Dense{a, b, c, c, b, a}

	for _, input := range inputs {
		hPre, softmax := MidProcess(h, input)

		switch input {
		case a:
			fmt.Println("入力: a")
		case b:
			fmt.Println("入力: b")
		case c:
			fmt.Println("入力: c")
		}

		fmt.Println("a:", softmax[0]*100, "%")
		fmt.Println("b:", softmax[1]*100, "%")
		fmt.Println("c:", softmax[2]*100, "%")

		fmt.Println("=====================================")

		h = hPre
	}
}

func MidProcess(h *mat.Dense, abc *mat.Dense) (*mat.Dense, []float64) {
	// 隠れ層
	h_1_pre := mat.NewDense(2, 1, nil)
	h_1_pre.Mul(w_x, abc)

	h_1_aft := mat.NewDense(2, 1, nil)
	h_1_aft.Mul(w_h, h)

	h_1_pre_plus_aft := mat.NewDense(2, 1, nil)
	h_1_pre_plus_aft.Add(h_1_pre, h_1_aft)

	h_1_pre_tanh := mat.NewDense(2, 1, nil)
	h_1_pre_tanh.Add(h_1_pre_plus_aft, b_h)

	h_1_aft_tanh := Tanh(h_1_pre_tanh.RawMatrix().Data)
	h_1 := mat.NewDense(2, 1, h_1_aft_tanh)

	// 出力層
	logits_pre := mat.NewDense(3, 1, nil)
	logits_pre.Mul(w_y, h_1)

	logits := mat.NewDense(3, 1, nil)
	logits.Add(logits_pre, b_y)

	// ソフトマックス関数を適用
	softmax := Softmax(logits.RawMatrix().Data)

	return h_1, softmax
}

func Softmax(a []float64) []float64 {
	sum := 0.0
	result := make([]float64, len(a))
	for _, val := range a {
		sum += math.Exp(val)
	}
	for i, val := range a {
		result[i] = math.Exp(val) / sum
	}
	return result
}

func Tanh(a []float64) []float64 {
	result := make([]float64, len(a))
	for i, val := range a {
		result[i] = math.Tanh(val)
	}
	return result
}
