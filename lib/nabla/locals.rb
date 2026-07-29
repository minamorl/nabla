# frozen_string_literal: true

module Nabla
  # ==================================================================
  # Locals — k.let 束縛を跨いで微分するための橋。
  #
  # カーネル本体は中間値を k.let で束ねるので、式の中には Local ノードが
  # 現れる。その定義は式木ではなく**文のリスト**にある。微分器は文を
  # 見ないので、定義表 (env) を渡してやる必要がある。
  #
  # 掟がひとつある: **代入される局所変数は env に入れない**。
  # `k.assign` で書き換わる変数はループ横断の状態であって純粋な束縛では
  # ないので、初期値を展開すると黙って別の関数を微分することになる。
  # 展開できないものは env から外し、触れられたら Opaque で落とす。
  # ==================================================================
  module Locals
    module_function

    # 文のリストから { Local => 定義式 } を作る。ループ本体・分岐本体にも
    # 降りる。代入される局所変数は除外する (上記の掟)。
    def environment(statements)
      definitions = {}
      assigned = {}
      scan(statements, definitions, assigned)
      assigned.each_key { |local| definitions.delete(local) }
      definitions
    end

    def scan(statements, definitions, assigned)
      statements.each do |statement|
        case statement
        when Moissanite::Let then definitions[statement.local] = statement.expr
        when Moissanite::Assign then assigned[statement.local] = true
        when Moissanite::Count then scan(statement.body, definitions, assigned)
        when Moissanite::If
          scan(statement.then_stmts, definitions, assigned)
          scan(statement.else_stmts, definitions, assigned)
        end
      end
    end

    # 直線的な本体 (Let と Ret のみ) を 1 本の式へ畳む。
    #
    # 展開しても式が膨らまないのは、戻り値が DAG だからである。let は
    # 「共有ノード」になるだけで複製されない。そして Emitter がそれを
    # また let に戻す。往復して元に戻る。
    #
    # 代入とループを含む本体は畳めない (状態が時間方向に変化するので、
    # 静的な式には潰れない)。それらは Nabla.local_env を使い、ループ本体
    # の中で勾配を組む方式を採る (README の縮約ループの節を参照)。
    def flatten(statements)
      environment = {}
      result = nil
      statements.each do |statement|
        case statement
        when Moissanite::Let then environment[statement.local] = substitute(statement.expr, environment)
        when Moissanite::Ret then result = substitute(statement.expr, environment)
        else raise Unsupported, "flatten handles straight-line bodies (Let/Ret) only, got #{statement.class}"
        end
      end
      raise Unsupported, 'body has no ret' unless result

      result
    end

    def substitute(node, environment)
      return environment.fetch(node) if node.is_a?(Moissanite::Local) && environment.key?(node)

      case node
      when Moissanite::BinOp
        Moissanite::Expr.arith(node.op, substitute(node.lhs, environment), substitute(node.rhs, environment))
      when Moissanite::Neg then -substitute(node.expr, environment)
      when Moissanite::Cast then Moissanite::Expr.cast(node.type, substitute(node.expr, environment))
      when Moissanite::MathOp then substitute_math(node, environment)
      when Moissanite::Select
        Moissanite::Expr.select(
          substitute(node.cond, environment),
          substitute(node.then_e, environment),
          substitute(node.else_e, environment)
        )
      else node
      end
    end

    def substitute_math(node, environment)
      args = node.args.map { |arg| substitute(arg, environment) }
      args.size == 1 ? Moissanite::Expr.math1(node.fn, args[0]) : Moissanite::Expr.math2(node.fn, *args)
    end
  end
end
