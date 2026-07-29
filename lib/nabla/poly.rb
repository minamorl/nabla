# frozen_string_literal: true

module Nabla
  # ==================================================================
  # Poly — 式をひとつの変数についての多項式として見る試み。
  #
  # 返すのは { 次数 => 係数 }、多項式として見えなければ **nil**。
  # 「見えなかった」を「0 次多項式だった」に丸めないことが要点で、
  # 丸めると積分規則が静かに間違った答えを出す。
  #
  # moissanite に冪演算は無いので x^3 は x*x*x である。それを次数に
  # 畳むのがここの仕事で、これがあるから多項式の積分が厳密になる。
  # ==================================================================
  module Poly
    module_function

    def of(expression, variable)
      case expression
      when Moissanite::Const then { 0 => expression.value }
      when Moissanite::Param, Moissanite::Load then Rewrite.same?(expression, variable) ? { 1 => 1.0 } : nil
      when Moissanite::Neg then of(expression.expr, variable)&.transform_values(&:-@)
      when Moissanite::BinOp then binop(expression, variable)
      end
    end

    def binop(node, variable)
      left = of(node.lhs, variable)
      right = of(node.rhs, variable)
      return nil unless left && right

      case node.op
      when :+ then merge(left, right)
      when :- then merge(left, right.transform_values(&:-@))
      when :* then multiply(left, right)
      when :/ then divide(left, right)
      end
    end

    def merge(left, right)
      left.merge(right) { |_, a, b| a + b }
    end

    def multiply(left, right)
      product = Hash.new(0.0)
      left.each { |i, a| right.each { |j, b| product[i + j] += a * b } }
      product
    end

    # 多項式で割れるのは定数で割るときだけ。x/(x+1) は多項式ではない。
    def divide(left, right)
      return nil unless right.keys == [0] && !right[0].zero?

      left.transform_values { |c| c / right[0] }
    end

    # { 次数 => 係数 } を式に戻す。係数 0 の項は落とす。
    def to_expression(coefficients, variable)
      coefficients.reject { |_, c| c.zero? }.sort.reverse
                  .inject(Simplify.number(0.0)) do |acc, (degree, coefficient)|
        Simplify.add(acc, term(degree, coefficient, variable))
      end
    end

    def term(degree, coefficient, variable)
      return Simplify.number(coefficient) if degree.zero?

      (0...degree).inject(Simplify.number(coefficient)) { |acc, _| Simplify.multiply(acc, variable) }
    end

    # 一次式かどうか (係数 a != 0 の a*v + b)。sin/cos/exp の引数がこの形の
    # ときだけ置換 u = a*v + b が厳密に効くので、その判定に使う。
    def linear(expression, variable)
      coefficients = of(expression, variable)
      return nil unless coefficients

      slope = coefficients[1]
      return nil if slope.nil? || slope.zero?
      return nil unless (coefficients.keys - [0, 1]).empty?

      [slope, coefficients[0] || 0.0]
    end
  end
end
