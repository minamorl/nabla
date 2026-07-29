# frozen_string_literal: true

require_relative 'test_helper'

# k.let 束縛を跨ぐ微分と、跨げないときの振る舞い。
#
# ここで一番大事なのは速度でも表現力でもなく、**黙って 0 を返さないこと**。
# 勾配 0 は「その入力は結果に影響しない」という主張であり、知らないことを
# 主張に変えるのは自動微分で最も見つけにくい間違いだからである。
class LocalsTest < Minitest::Test
  include NablaTest::Helpers

  def build_body
    builder = Moissanite::Builder.new
    x = Moissanite::Param.new(type: :f64, name: :x, index: 0)
    squared = builder.let(x * x)
    shifted = builder.let(squared + 3.0)
    logged = builder.let(shifted.log)
    builder.ret(logged * squared)
    [builder, x]
  end

  # f = log(x^2 + 3) * x^2
  # f' = 2x*log(x^2+3) + x^2 * 2x/(x^2+3)
  def analytic(x)
    (2 * x * Math.log((x * x) + 3)) + ((x * x) * 2 * x / ((x * x) + 3))
  end

  def test_opaque_local_raises_instead_of_returning_a_silent_zero
    builder = Moissanite::Builder.new
    x = Moissanite::Param.new(type: :f64, name: :x, index: 0)
    bound = builder.let(x * x)

    error = assert_raises(Nabla::Opaque) { Nabla.gradient(bound + 1.0, [x]) }
    assert_match(/k\.let binding/, error.message)
    assert_match(/env:/, error.message)
  end

  def test_supplying_the_environment_differentiates_through_the_binding
    builder = Moissanite::Builder.new
    x = Moissanite::Param.new(type: :f64, name: :x, index: 0)
    bound = builder.let(x * x)
    env = Nabla.local_env(builder.statements)

    kernel = Moissanite.kernel(:through_let, x: :f64) do |k, param|
      inner = Moissanite::Builder.new
      squared = inner.let(param * param)
      scope = Nabla.local_env(inner.statements)
      partial = Nabla.gradient(squared + 1.0, [param], env: scope).fetch(param)
      k.ret Nabla.emit(k, partial, env: scope)
    end

    assert_in_delta 6.0, kernel.call(3.0), 0.0
    refute_empty env
    refute_nil bound
  end

  def test_flatten_turns_a_straight_line_body_into_one_expression
    builder, x = build_body
    flat = Nabla.flatten(builder.statements)

    # 展開しても膨らまないのは戻り値が DAG だから — let は共有ノードに
    # なるだけで複製されない。
    assert_operator Nabla.size(flat), :<, Nabla.tree_size(flat)

    kernel = Moissanite.kernel(:flattened, x: :f64) do |k, param|
      inner = Moissanite::Builder.new
      squared = inner.let(param * param)
      shifted = inner.let(squared + 3.0)
      inner.ret(inner.let(shifted.log) * squared)
      partial = Nabla.gradient(Nabla.flatten(inner.statements), [param]).fetch(param)
      k.ret Nabla.emit(k, partial)
    end

    [0.7, 1.9, -2.3].each do |value|
      assert_in_delta analytic(value), kernel.call(value), 0.0,
                      "derivative through k.let must match the analytic solution exactly at x=#{value}"
    end
    refute_nil x
  end

  # k.assign される局所変数はループ横断の状態であって純粋な束縛ではない。
  # 初期値を展開すると別の関数を微分することになるので、env から外す。
  def test_reassigned_locals_are_excluded_from_the_environment
    builder = Moissanite::Builder.new
    pure = builder.let(1.5)
    mutated = builder.let(0.0)
    builder.assign(mutated, mutated + 2.0)
    env = Nabla.local_env(builder.statements)

    assert env.key?(pure), 'a pure binding belongs in the environment'
    refute env.key?(mutated), 'loop-carried state must not be inlined as if it were its initial value'
  end

  def test_flatten_refuses_bodies_it_cannot_fold
    builder = Moissanite::Builder.new
    acc = builder.let(0.0)
    builder.count(4) { builder.assign(acc, acc + 1.0) }
    builder.ret acc

    assert_raises(Nabla::Unsupported) { Nabla.flatten(builder.statements) }
  end
end
