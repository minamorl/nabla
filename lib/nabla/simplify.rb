# frozen_string_literal: true

module Nabla
  # ==================================================================
  # Simplify — 定数畳み込みと単位元の除去。
  #
  # 微分は構文変換なので、規則をそのまま適用すると `1.0 * x + x * 1.0`
  # のような残骸が必ず出る。放置しても gcc が畳んでくれるが、それでは
  # **式を読めない**し、積分の規則照合もできない (`x * 1.0` は多項式
  # として認識されるべきだが、照合器が単位元を知らなければ落ちる)。
  #
  # 高階微分では効きが積み上がる。実測で微分結果が 40% 縮む。
  #
  # ここは代数的正規形ではない。同類項の収集や展開はしないので、
  # `x + x` は畳むが `2*x + 3*x` は畳まない。**畳めなかったものを
  # 畳んだと偽らない**ことのほうが、賢さより重要である。
  # ==================================================================
  module Simplify
    module_function

    def run(expression, memo = {})
      return memo[expression] if memo.key?(expression)

      memo[expression] = case expression
                         when Moissanite::BinOp
                           binop(expression.op, run(expression.lhs, memo), run(expression.rhs, memo))
                         when Moissanite::Neg then negate(run(expression.expr, memo))
                         when Moissanite::MathOp then math(expression.fn, expression.args.map { |a| run(a, memo) })
                         else expression
                         end
    end

    def constant?(node) = node.is_a?(Moissanite::Const)
    def value_of(node) = node.value
    def number(value) = Moissanite::Expr.lift(value.to_f, :f64)

    def negate(node)
      return number(-value_of(node)) if constant?(node)
      return node.expr if node.is_a?(Moissanite::Neg)

      -node
    end

    def binop(op, lhs, rhs)
      return fold(op, lhs, rhs) if constant?(lhs) && constant?(rhs)

      case op
      when :+ then add(lhs, rhs)
      when :- then subtract(lhs, rhs)
      when :* then multiply(lhs, rhs)
      when :/ then divide(lhs, rhs)
      else Moissanite::Expr.arith(op, lhs, rhs)
      end
    end

    # 0 除算は畳まない。moissanite の意味論では oracle が MathError を
    # 上げ native は未定義なので、ここで勝手に Inf/NaN を作らずそのまま
    # 式として残し、実行時の規則に委ねる。
    def fold(op, lhs, rhs)
      left = value_of(lhs)
      right = value_of(rhs)
      case op
      when :+ then number(left + right)
      when :- then number(left - right)
      when :* then number(left * right)
      when :/ then right.zero? ? Moissanite::Expr.arith(op, lhs, rhs) : number(left / right)
      else Moissanite::Expr.arith(op, lhs, rhs)
      end
    end

    def add(lhs, rhs)
      return rhs if zero?(lhs)
      return lhs if zero?(rhs)
      return multiply(number(2.0), lhs) if lhs == rhs

      lhs + rhs
    end

    def subtract(lhs, rhs)
      return negate(rhs) if zero?(lhs)
      return lhs if zero?(rhs)
      return number(0.0) if lhs == rhs

      lhs - rhs
    end

    def multiply(lhs, rhs)
      return number(0.0) if zero?(lhs) || zero?(rhs)
      return rhs if one?(lhs)
      return lhs if one?(rhs)
      return negate(rhs) if minus_one?(lhs)
      return negate(lhs) if minus_one?(rhs)

      # 定数を左へ寄せる弱い正規化。完全な順序付けではないが、これだけで
      # 多項式照合の当たりがかなり良くなる。
      constant?(rhs) && !constant?(lhs) ? rhs * lhs : lhs * rhs
    end

    def divide(lhs, rhs)
      return lhs if one?(rhs)
      return number(0.0) if zero?(lhs) && !zero?(rhs)
      return number(1.0) if lhs == rhs && !zero?(lhs)

      lhs / rhs
    end

    # 定義域の外 (負数の log / sqrt) は畳まない。C と Ruby で NaN の出方が
    # 揃う保証をここで作る必要はなく、式のまま残せば oracle が正典になる。
    def math(fn, args)
      return build_math(fn, args) unless args.all? { |a| constant?(a) }

      values = args.map { |a| value_of(a) }
      folded = fold_math(fn, values)
      folded ? number(folded) : build_math(fn, args)
    end

    def fold_math(fn, values)
      case fn
      when :sin then Math.sin(values[0])
      when :cos then Math.cos(values[0])
      when :exp then Math.exp(values[0])
      when :abs then values[0].abs
      when :min then values.min
      when :max then values.max
      when :sqrt then values[0].negative? ? nil : Math.sqrt(values[0])
      when :log then values[0] <= 0 ? nil : Math.log(values[0])
      end
    end

    def build_math(fn, args)
      args.size == 1 ? Moissanite::Expr.math1(fn, args[0]) : Moissanite::Expr.math2(fn, *args)
    end

    def zero?(node) = constant?(node) && value_of(node).zero?
    def one?(node) = constant?(node) && value_of(node) == 1.0
    def minus_one?(node) = constant?(node) && value_of(node) == -1.0
  end
end
