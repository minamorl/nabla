# frozen_string_literal: true

module Nabla
  # ==================================================================
  # Rewrite — 式木の置換と依存判定。積分・極限・検証がどれも
  # 「変数を何かで置き換える」を必要とするので、ここに集約する。
  # ==================================================================
  module Rewrite
    module_function

    # e の中の variable を replacement で置き換える。置換のたびに簡約を
    # かけるのは、定数を入れた瞬間に畳めるものが大量に生まれるため
    # (定積分の F(b) - F(a) がまさにそれ)。
    def substitute(expression, variable, replacement, memo = {})
      return replacement if same?(expression, variable)
      return memo[expression] if memo.key?(expression)

      memo[expression] = rebuild(expression, variable, replacement, memo)
    end

    def rebuild(node, variable, replacement, memo)
      case node
      when Moissanite::BinOp
        Simplify.binop(node.op,
                       substitute(node.lhs, variable, replacement, memo),
                       substitute(node.rhs, variable, replacement, memo))
      when Moissanite::Neg then Simplify.negate(substitute(node.expr, variable, replacement, memo))
      when Moissanite::MathOp then rebuild_math(node, variable, replacement, memo)
      when Moissanite::Select
        Moissanite::Expr.select(
          substitute(node.cond, variable, replacement, memo),
          substitute(node.then_e, variable, replacement, memo),
          substitute(node.else_e, variable, replacement, memo)
        )
      else node
      end
    end

    def rebuild_math(node, variable, replacement, memo)
      args = node.args.map { |arg| substitute(arg, variable, replacement, memo) }
      Simplify.math(node.fn, args)
    end

    # Param は名前で、それ以外 (Load など) は構造で同一性を見る。
    def same?(node, variable)
      if node.is_a?(Moissanite::Param) && variable.is_a?(Moissanite::Param)
        node.name == variable.name
      else
        node == variable
      end
    end

    # e が variable に依存しないか。積分の定数倍規則などが依存する述語で、
    # **偽を返すほうが安全側**である (依存しているのに「しない」と答えると
    # 定数として括り出してしまい、答えが静かに間違う)。
    def free_of?(expression, variable, memo = {})
      return false if same?(expression, variable)
      return memo[expression] if memo.key?(expression)

      memo[expression] = DAG.children(expression).all? { |child| free_of?(child, variable, memo) }
    end
  end
end
