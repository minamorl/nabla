# frozen_string_literal: true

module Nabla
  # ==================================================================
  # Emitter — 共有部分式を k.let に束ねながら式を放出する。
  #
  # **これは最適化ではなく必須の部品である。** 逆モードは構造上、
  # 大量に共有された DAG を作る (全ての偏微分が同じ中間量を参照する)。
  # 共有を保ったまま放出しなければ、moissanite の式木は木なので DAG が
  # 木へ展開され、逆モードで得た O(N) が O(N^2) に戻ってしまう。
  #
  # 実測: 50 入力の勾配は DAG で 460 ノード、木に展開すると 51,959
  # ノード (113 倍)。ここを通すか通さないかで利得が消し飛ぶ。
  #
  # 2 回以上参照される非自明なノードだけを束ねる。定数・パラメータ・
  # 既存の局所変数は束ねても無意味なので素通しする。
  # ==================================================================
  class Emitter
    def initialize(builder, env: nil)
      @builder = builder
      @env = env
      @uses = Hash.new(0)
      @memo = {}
    end

    # 放出する式の集合を先に見せて参照回数を数える。emit の前に必ず呼ぶ
    # (回数が判らないと何を束ねるべきか決まらない)。
    def prepare(expressions)
      seen = {}
      stack = Array(expressions)
      until stack.empty?
        node = stack.pop
        @uses[node] += 1
        next if seen.key?(node)

        seen[node] = true
        stack.concat(DAG.children(node, @env))
      end
      self
    end

    # 式を builder へ放出し、参照できる値 (Local または式) を返す。
    # 同じノードは 2 度組み立てられない (memo)。
    def emit(expression)
      return @memo[expression] if @memo.key?(expression)

      value = rebuild(expression)
      value = @builder.let(value) if bind?(expression)
      @memo[expression] = value
      value
    end

    private

    def bind?(node)
      return false if node.is_a?(Moissanite::Const)
      return false if node.is_a?(Moissanite::Param) || node.is_a?(Moissanite::Local)
      return false if node.is_a?(Moissanite::Load)

      @uses[node] > 1
    end

    # 子を先に放出してから自分を組み直す。子は memo 済みなので、共有は
    # そのまま Local の共有として残る。
    def rebuild(node)
      case node
      when Moissanite::BinOp then Moissanite::Expr.arith(node.op, emit(node.lhs), emit(node.rhs))
      when Moissanite::Neg then -emit(node.expr)
      when Moissanite::Cast then Moissanite::Expr.cast(node.type, emit(node.expr))
      when Moissanite::MathOp then rebuild_math(node)
      when Moissanite::Select
        Moissanite::Expr.select(emit(node.cond), emit(node.then_e), emit(node.else_e))
      when Moissanite::Local then @env ? emit(DAG.local_definition(node, @env)) : node
      else node
      end
    end

    def rebuild_math(node)
      args = node.args.map { |arg| emit(arg) }
      args.size == 1 ? Moissanite::Expr.math1(node.fn, args[0]) : Moissanite::Expr.math2(node.fn, *args)
    end
  end
end
