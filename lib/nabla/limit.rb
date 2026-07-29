# frozen_string_literal: true

module Nabla
  # ==================================================================
  # Limit — L'Hopital の規則による商の極限。
  #
  # ここは**ほぼ何も書いていない**。0/0 や ∞/∞ を見つけたら分子と分母を
  # 微分して評価し直すだけで、その微分は既に Forward が持っている。
  # 微分を式として返す設計にした結果、極限が副産物として出てくる。
  #
  # 評価は native カーネル経由の数値評価である。したがって「不定形か
  # どうか」の判定は閾値つきの数値判断であり、記号的な判定ではない。
  # 極めて小さい非ゼロ値を 0 と見なして誤った極限を返しうるので、
  # ここは **CAS 全体の中で最も弱い部分**だと明示しておく。
  #
  # 級数展開 (Gruntz) による判定は未実装。L'Hopital が回数内に解けない
  # 形は NotElementary で断る。
  # ==================================================================
  module Limit
    ZERO_THRESHOLD = 1e-12

    module_function

    # lim_{v -> point} numerator / denominator
    def quotient(numerator, denominator, variable, point, applications: 8)
      applications.times do
        top = evaluate(numerator, variable, point)
        bottom = evaluate(denominator, variable, point)
        return top / bottom unless indeterminate?(top, bottom)

        numerator = Simplify.run(Forward.derivative(numerator, variable))
        denominator = Simplify.run(Forward.derivative(denominator, variable))
      end
      raise NotElementary, "limit did not resolve within #{applications} applications of L'Hopital"
    end

    def indeterminate?(top, bottom)
      return true if top.abs < ZERO_THRESHOLD && bottom.abs < ZERO_THRESHOLD
      return true if !top.finite? && !bottom.finite?

      false
    end

    # 一点での数値評価。式を native カーネルに落として呼ぶので、
    # oracle と同じ意味論で評価される。
    def evaluate(expression, variable, point)
      Moissanite.kernel(:limit_point, x: :f64) do |builder, x|
        builder.ret Nabla.emit(builder, Rewrite.substitute(expression, variable, x))
      end.call(point)
    end
  end
end
