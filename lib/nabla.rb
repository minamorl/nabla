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
require_relative 'nabla/simplify'
require_relative 'nabla/rewrite'
require_relative 'nabla/poly'
require_relative 'nabla/integrate'
require_relative 'nabla/verify'
require_relative 'nabla/limit'

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

  # ---- 記号代数層 --------------------------------------------------------

  # 定数畳み込みと単位元の除去。代数的正規形ではない (同類項は集めない)。
  def simplify(expression)
    Simplify.run(expression)
  end

  # 不定積分。解けない形は NotElementary で断る — 近似は返さない。
  def integrate(expression, variable)
    Integrate.antiderivative(expression, variable)
  end

  # 定積分 F(hi) - F(lo)。内側の結果が式として返るので、重積分は
  # これを入れ子にするだけで済む。
  def definite_integral(expression, variable, lower, upper)
    Integrate.definite(expression, variable, lower, upper)
  end

  # 積分結果を微分し戻して検証する。:symbolic なら微分規則が正しい限り
  # 正しく、:numeric なら反例が見つからなかったことしか言えない。
  def verify_integral(integrand, primitive, variable, **)
    Verify.integral(integrand, primitive, variable, **)
  end

  # L'Hopital による商の極限。不定形の判定が数値なので、この CAS で
  # 最も弱い部分である (Limit の注記を参照)。
  def limit(numerator, denominator, variable, at:, applications: 8)
    Limit.quotient(numerator, denominator, variable, at, applications: applications)
  end

  # 変数を式で置き換える。
  def substitute(expression, variable, replacement)
    Rewrite.substitute(expression, variable, replacement)
  end

  # 式が変数に依存しないか。
  def free_of?(expression, variable)
    Rewrite.free_of?(expression, variable)
  end
end
