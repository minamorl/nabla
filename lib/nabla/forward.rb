# frozen_string_literal: true

module Nabla
  # ==================================================================
  # Forward — 前進モード微分。1 回の走査で **1 つの**入力に対する
  # 導関数を返す。
  #
  # 入力が多いときは逆モードに劣る (入力ごとに 1 走査なので合計
  # O(N^2))。それでも前進モードを持つ理由は二つ:
  #
  #   - **高階微分**。返り値が式なので、そのまま次の階へ渡せる。
  #     d3 = d(d(d(f))) が素直に書ける。
  #   - **逆モードの検証**。両者は数学的に同じ値を出すはずで、実装が
  #     独立しているので相互検証になる (test/cross_mode_test.rb)。
  # ==================================================================
  module Forward
    module_function

    # node を target で微分した式を返す。target は Param か Load。
    def derivative(node, target, env: nil)
      case node
      when Moissanite::Const then Moissanite::Expr.lift(0.0, :f64)
      when Moissanite::Param, Moissanite::Load then seed_for(node, target)
      when Moissanite::Local then derivative(DAG.local_definition(node, env), target, env: env)
      when Moissanite::Neg then -derivative(node.expr, target, env: env)
      when Moissanite::BinOp then binop(node, target, env)
      when Moissanite::MathOp then mathop(node, target, env)
      when Moissanite::Select then select(node, target, env)
      else raise Unsupported, "cannot differentiate #{node.class}"
      end
    end

    # 自分自身なら 1、それ以外の独立変数なら 0。Param は名前で、Load は
    # 構造 (バッファと添字式) で同一性を見る。
    def seed_for(node, target)
      same =
        if node.is_a?(Moissanite::Param) && target.is_a?(Moissanite::Param)
          node.name == target.name
        else
          node == target
        end
      Moissanite::Expr.lift(same ? 1.0 : 0.0, :f64)
    end

    def binop(node, target, env)
      lhs = node.lhs
      rhs = node.rhs
      dl = derivative(lhs, target, env: env)
      dr = derivative(rhs, target, env: env)
      case node.op
      when :+ then dl + dr
      when :- then dl - dr
      when :* then (dl * rhs) + (lhs * dr)
      when :/ then ((dl * rhs) - (lhs * dr)) / (rhs * rhs)
      else raise Unsupported, "no derivative rule for operator #{node.op}"
      end
    end

    def mathop(node, target, env)
      operand = node.args[0]
      inner = derivative(operand, target, env: env)
      case node.fn
      when :sin then operand.cos * inner
      when :cos then -operand.sin * inner
      when :exp then operand.exp * inner
      when :log then inner / operand
      when :sqrt then inner / (2.0 * operand.sqrt)
      when :abs then Moissanite::Expr.select(operand < 0.0, -1.0, 1.0) * inner
      when :min, :max then minmax(node, target, env)
      else raise Unsupported, "no derivative rule for #{node.fn}"
      end
    end

    def minmax(node, target, env)
      lhs, rhs = node.args
      takes_lhs = node.fn == :min ? (lhs < rhs) : (lhs > rhs)
      Moissanite::Expr.select(takes_lhs, derivative(lhs, target, env: env), derivative(rhs, target, env: env))
    end

    def select(node, target, env)
      Moissanite::Expr.select(
        node.cond, derivative(node.then_e, target, env: env), derivative(node.else_e, target, env: env)
      )
    end
  end
end
