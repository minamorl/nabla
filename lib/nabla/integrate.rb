# frozen_string_literal: true

module Nabla
  # ==================================================================
  # Integrate — 不定積分・定積分・重積分。
  #
  # 微分と違って積分は **合成的でない**。連鎖律に当たるものが無いので、
  # 「どんな式でも規則を下へ配れば答えが出る」という構造が使えない。
  # 初等関数の完全な判定には Risch アルゴリズムが要り、それを完全実装
  # した CAS は存在しない。
  #
  # だからここは「解ける形だけを解き、**解けない形は必ず断る**」方針を
  # 採る。断るべきところで近似を返すのが最悪で、それは呼び手から見て
  # 成功と区別がつかないからである。扱うのは:
  #
  #   - 多項式 (Poly が次数に畳めるもの)
  #   - 線形性 (和・差・定数倍・定数での除算)
  #   - 一次式を引数に取る sin / cos / exp
  #   - 1/(a*v + b)
  #   - 被積分変数に依存しない式 (定数の積分)
  #
  # 部分積分・部分分数分解・一般の置換は未実装で、NotElementary を返す。
  # ==================================================================
  module Integrate
    module_function

    # 不定積分 (積分定数は付けない)。解けなければ NotElementary。
    def antiderivative(expression, variable)
      Simplify.run(compute(Simplify.run(expression), variable))
    end

    def compute(expression, variable)
      coefficients = Poly.of(expression, variable)
      return integrate_polynomial(coefficients, variable) if coefficients
      return Simplify.multiply(expression, variable) if Rewrite.free_of?(expression, variable)

      case expression
      when Moissanite::Neg then Simplify.negate(compute(expression.expr, variable))
      when Moissanite::BinOp then binop(expression, variable)
      when Moissanite::MathOp then mathop(expression, variable)
      else raise NotElementary, "no integration rule for #{expression.class}"
      end
    end

    # sum c_d v^d  ->  sum c_d/(d+1) v^(d+1)
    def integrate_polynomial(coefficients, variable)
      Poly.to_expression(coefficients.to_h { |degree, c| [degree + 1, c / (degree + 1)] }, variable)
    end

    def binop(node, variable)
      case node.op
      when :+ then Simplify.add(compute(node.lhs, variable), compute(node.rhs, variable))
      when :- then Simplify.subtract(compute(node.lhs, variable), compute(node.rhs, variable))
      when :* then product(node, variable)
      when :/ then quotient(node, variable)
      else raise NotElementary, "no integration rule for #{node.op}"
      end
    end

    # 定数倍だけを扱う。両辺が変数に依存する積は部分積分が要るので断る。
    def product(node, variable)
      return Simplify.multiply(node.lhs, compute(node.rhs, variable)) if Rewrite.free_of?(node.lhs, variable)
      return Simplify.multiply(node.rhs, compute(node.lhs, variable)) if Rewrite.free_of?(node.rhs, variable)

      raise NotElementary, 'product of two factors that both depend on the variable needs integration by parts'
    end

    def quotient(node, variable)
      return Simplify.divide(compute(node.lhs, variable), node.rhs) if Rewrite.free_of?(node.rhs, variable)

      slope, = Poly.linear(node.rhs, variable)
      unless Rewrite.free_of?(node.lhs, variable) && slope
        raise NotElementary, 'quotient with a variable denominator needs partial fractions'
      end

      # c/(a*v + b) -> (c/a) * log(a*v + b)
      Simplify.multiply(Simplify.divide(node.lhs, Simplify.number(slope)), node.rhs.log)
    end

    # 引数が一次式のときだけ、置換 u = a*v + b が厳密に効く。
    # 非線形な引数 (sin(x^2) など) は初等関数で書けないことすらあるので断る。
    def mathop(node, variable)
      operand = node.args[0]
      slope, = Poly.linear(operand, variable)
      raise NotElementary, "#{node.fn} of a non-linear argument" unless slope

      inverse = Simplify.number(1.0 / slope)
      case node.fn
      when :sin then Simplify.multiply(Simplify.negate(inverse), operand.cos)
      when :cos then Simplify.multiply(inverse, operand.sin)
      when :exp then Simplify.multiply(inverse, operand.exp)
      else raise NotElementary, "no integration rule for #{node.fn}"
      end
    end

    # 定積分 F(hi) - F(lo)。lo/hi は数でも式でもよい。
    #
    # 重積分に特別扱いは要らない: 内側の結果が **式** として返るので、
    # それをそのまま外側の被積分関数に渡せばよい。積分が合成でないのは
    # 「規則が」であって、定積分の入れ子は素直に合成する。
    def definite(expression, variable, lower, upper)
      primitive = antiderivative(expression, variable)
      at_upper = Rewrite.substitute(primitive, variable, lift(upper))
      at_lower = Rewrite.substitute(primitive, variable, lift(lower))
      Simplify.run(Simplify.subtract(at_upper, at_lower))
    end

    def lift(bound)
      bound.is_a?(Numeric) ? Simplify.number(bound) : bound
    end
  end
end
