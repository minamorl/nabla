# frozen_string_literal: true

module Nabla
  # ==================================================================
  # Reverse — 逆モード自動微分。**実行時テープを持たない**。
  #
  # 普通の逆モード AD は、実行しながら演算を記録し (テープ)、それを
  # 逆走して随伴を積む。テープは実行時のメモリと分岐を食う。
  #
  # ここではその必要が無い。プログラムが**データであり、かつ静的**
  # だからである。式は組んだ瞬間に全体が判っているので、逆走はカーネル
  # 構築時の DAG 走査になる。記録すべき実行が存在しない。
  #
  # 結果として N 入力の全偏微分が **1 回の走査**で出る。前進モードは
  # 入力ごとに 1 走査なので合計 O(N^2) ノード、逆モードは O(N)。
  # ==================================================================
  module Reverse
    module_function

    # root を targets の各要素で偏微分した式を返す ({ target => Expr })。
    # targets は Param か Load (バッファ読み) のノード。
    def gradient(root, targets, env: nil)
      adjoint = { root => Moissanite::Expr.lift(1.0, :f64) }
      accumulate = lambda do |node, contribution|
        current = adjoint[node]
        adjoint[node] = current ? current + contribution : contribution
      end

      DAG.topological(root, env).each do |node|
        seed = adjoint[node]
        next unless seed # root から到達しない枝には随伴が無い

        push(node, seed, accumulate, env)
      end

      targets.to_h { |target| [target, adjoint[target] || Moissanite::Expr.lift(0.0, :f64)] }
    end

    # 随伴規則: d(root)/d(node) == seed が判っているとき、各子へ寄与を押す。
    def push(node, seed, add, env = nil)
      case node
      when Moissanite::Local then add.call(DAG.local_definition(node, env), seed)
      when Moissanite::BinOp then push_binop(node, seed, add)
      when Moissanite::Neg then add.call(node.expr, -seed)
      when Moissanite::MathOp then push_math(node, seed, add)
      when Moissanite::Select then push_select(node, seed, add)
      end
    end

    def push_binop(node, seed, add)
      lhs = node.lhs
      rhs = node.rhs
      case node.op
      when :+ then (add.call(lhs, seed)
                    add.call(rhs, seed))
      when :- then (add.call(lhs, seed)
                    add.call(rhs, -seed))
      when :* then (add.call(lhs, seed * rhs)
                    add.call(rhs, seed * lhs))
      when :/ then (add.call(lhs, seed / rhs)
                    add.call(rhs, -(seed * lhs) / (rhs * rhs)))
      else raise Unsupported, "no adjoint rule for operator #{node.op}"
      end
    end

    def push_math(node, seed, add)
      operand = node.args[0]
      case node.fn
      when :sin then add.call(operand, seed * operand.cos)
      when :cos then add.call(operand, -(seed * operand.sin))
      when :exp then add.call(operand, seed * operand.exp)
      when :log then add.call(operand, seed / operand)
      when :sqrt then add.call(operand, seed / (2.0 * operand.sqrt))
      when :abs then add.call(operand, seed * Moissanite::Expr.select(operand < 0.0, -1.0, 1.0))
      when :min, :max then push_minmax(node, seed, add)
      else raise Unsupported, "no adjoint rule for #{node.fn}"
      end
    end

    # 折れ点 (l == r) では微分が存在しない。fmin/fmax の意味論に合わせて
    # 「選ばれなかった側に 0」を配る片側の慣習を採る — 分岐が同点のとき
    # どちらへ流すかは規約であって定理ではない、と明示しておく。
    def push_minmax(node, seed, add)
      lhs, rhs = node.args
      zero = Moissanite::Expr.lift(0.0, :f64)
      takes_lhs = node.fn == :min ? (lhs < rhs) : (lhs > rhs)
      add.call(lhs, Moissanite::Expr.select(takes_lhs, seed, zero))
      add.call(rhs, Moissanite::Expr.select(takes_lhs, zero, seed))
    end

    # 選ばれた枝にだけ随伴が流れる。条件式自体は bool なので微分しない。
    def push_select(node, seed, add)
      zero = Moissanite::Expr.lift(0.0, :f64)
      add.call(node.then_e, Moissanite::Expr.select(node.cond, seed, zero))
      add.call(node.else_e, Moissanite::Expr.select(node.cond, zero, seed))
    end
  end
end
