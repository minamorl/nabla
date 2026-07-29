# frozen_string_literal: true

# ==================================================================
# nabla — moissanite の式木に対する記号的自動微分。
#
# moissanite のカーネルは「実行時に組み立てた検査可能な式木」である。
# 微分はその木に対する変換なので、微分結果もまた木になり、同じ経路で
# C へ落ちて -O3 でコンパイルされる。**導関数の計算はカーネル構築時に
# 終わっており、実行時コストはゼロ**である。
#
# 中心的な事実はひとつ: **プログラムがデータであり、かつ静的**。
# だから逆モード AD に実行時テープが要らない。記録すべき実行が無く、
# 逆走は構築時の DAG 走査に過ぎない。
# ==================================================================
require 'moissanite'

require_relative 'nabla/version'
require_relative 'nabla/errors'
require_relative 'nabla/dag'
require_relative 'nabla/forward'
require_relative 'nabla/reverse'
require_relative 'nabla/emitter'
require_relative 'nabla/locals'

module Nabla
  module_function

  # 全偏微分を 1 回の走査で返す ({ target => Expr })。入力が 2 つ以上なら
  # 常にこちらを使う — 前進モードは入力ごとに 1 走査になる。
  def gradient(expression, targets, env: nil)
    Reverse.gradient(expression, targets, env: env)
  end

  # 1 入力に対する導関数。返り値が式なので、そのまま次の階へ渡せば
  # 高階微分になる (d3 = derivative(derivative(derivative(f, x), x), x))。
  def derivative(expression, target, env: nil)
    Forward.derivative(expression, target, env: env)
  end

  # 勾配を組んで builder へ放出するまでを一息で行う。targets の順に
  # 並んだ「参照できる値」の配列が返る。
  #
  # Emitter を通すのが要点で、共有部分式が k.let に束ねられる。
  # これを省くと逆モードの O(N) が放出時に O(N^2) へ戻る。
  #
  #   Moissanite.kernel(:grad, g: :f64_buf, w: :f64_buf, n: :i64) do |k, g, w, n|
  #     loss = build_loss(w)
  #     Nabla.emit_gradient(k, loss, (0...8).map { |j| w[j] })
  #          .each_with_index { |partial, j| k.store(g, j, partial) }
  #     k.ret 0
  #   end
  def emit_gradient(builder, expression, targets, env: nil)
    partials = gradient(expression, targets, env: env)
    emitter = Emitter.new(builder, env: env).prepare(partials.values)
    targets.map { |target| emitter.emit(partials.fetch(target)) }
  end

  # 単一の式を共有を保ったまま放出する。
  def emit(builder, expression, env: nil)
    Emitter.new(builder, env: env).prepare([expression]).emit(expression)
  end

  # k.let 束縛の定義表。builder.statements を渡す。
  def local_env(statements)
    Locals.environment(statements)
  end

  # 直線的なカーネル本体を 1 本の式へ畳む。
  def flatten(statements)
    Locals.flatten(statements)
  end

  # DAG としての大きさ (共有を数える) と、木としての大きさ (共有を展開
  # する)。両者の開きがそのまま Emitter の必要性を示す。
  def size(expression, env: nil)
    DAG.size(expression, env)
  end

  def tree_size(expression, env: nil)
    DAG.tree_size(expression, env)
  end
end
