# frozen_string_literal: true

module Nabla
  # ==================================================================
  # DAG — moissanite の式木を「共有された有向非巡回グラフ」として歩く。
  #
  # moissanite の Expr ノードは Data なので、等価性は構造等価である。
  # つまり Hash のキーにすると **同型の部分式は自動的に同一視される**。
  # 微分にとってこれは事故ではなく利益で、共有された部分式は随伴を
  # 1 スロットに集約し、放出も 1 回で済む。
  # ==================================================================
  module DAG
    module_function

    # 子ノード。env は Local (k.let 束縛) の定義表。
    #
    # Local を葉として扱うと、そこから先の依存が見えなくなり **黙って
    # 勾配 0 が返る**。自動微分でこれは最悪の失敗様式 (誰も気づけない)
    # なので、定義が無ければ 0 を返さず必ず例外にする。
    #
    # Load は微分の葉である。値はメモリから来るのであって、添字は
    # アドレスであって微分可能量ではない。添字に降りるとループカウンタが
    # 導関数に混入する。
    def children(node, env = nil)
      case node
      when Moissanite::Local then [local_definition(node, env)]
      when Moissanite::BinOp then [node.lhs, node.rhs]
      when Moissanite::Neg, Moissanite::Cast, Moissanite::Not then [node.expr]
      when Moissanite::MathOp then node.args.to_a
      when Moissanite::Select then [node.cond, node.then_e, node.else_e]
      else []
      end
    end

    def local_definition(node, env)
      return env.fetch(node) if env&.key?(node)

      raise Opaque,
            "cannot differentiate through `#{node.name}`: it is a k.let binding whose definition " \
            'was not supplied. Pass env: Nabla.local_env(builder.statements), or build the primal ' \
            'as a plain expression and let Nabla::Emitter reintroduce the bindings.'
    end

    # 親が子より先に来る順序。DAG では DFS 後行順の逆順が位相順序に
    # なる。逆モードが必要とするのはまさにこれで、あるノードの随伴は
    # 「全ての親が寄与を押し込んだ後」にのみ確定する。
    def topological(root, env = nil)
      seen = {}
      order = []
      visit(root, env, seen, order)
      order.reverse
    end

    # 再帰ではなく明示スタックで歩く (深い式木でも Ruby の stack level
    # too deep を踏まない — moissanite が実行をトランポリンで守るのと
    # 同じ理由)。
    def visit(root, env, seen, order)
      stack = [[root, false]]
      until stack.empty?
        node, expanded = stack.pop
        if expanded
          order << node
          next
        end
        next if seen.key?(node)

        seen[node] = true
        stack.push([node, true])
        children(node, env).each { |child| stack.push([child, false]) }
      end
    end

    # 共有を数えたノード数 (DAG としての大きさ)。
    def size(node, env = nil)
      seen = {}
      order = []
      visit(node, env, seen, order)
      order.size
    end

    # 共有を展開したノード数 (木としての大きさ)。Emitter を通さずに
    # 放出した場合の膨張量が判る — 逆モードでは両者が 100 倍以上開く。
    def tree_size(node, env = nil)
      1 + children(node, env).sum { |child| tree_size(child, env) }
    end
  end
end
