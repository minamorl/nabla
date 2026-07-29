# frozen_string_literal: true

require_relative 'test_helper'

# 縮約ループを跨ぐ勾配 — **テープなし**。
#
# 和による縮約では、各反復の寄与に対する随伴がちょうど 1 である。だから
# 「損失の勾配 = 各項の勾配の和」がそのまま成立し、逆モードをループ本体に
# 適用して累積するだけで厳密な勾配が出る。ループを記録する必要がない。
#
# ML の損失は例外なくバッチ和なので、ここが通らなければ実用にならない。
class ReductionTest < Minitest::Test
  include NablaTest::Helpers

  DIMENSION = 6

  # loss = sum_i (w . x_i - y_i)^2
  def loss_kernel
    Moissanite.kernel(:loss, w: :f64_buf, x: :f64_buf, y: :f64_buf, n: :i64) do |k, w, x, y, n|
      acc = k.let(0.0)
      k.count(n) do |i|
        base = k.let(i * DIMENSION)
        residual = k.let(prediction(w, x, base) - y[i])
        k.assign(acc, acc + (residual * residual))
      end
      k.ret acc
    end
  end

  def gradient_kernel
    Moissanite.kernel(:loss_grad, g: :f64_buf, w: :f64_buf, x: :f64_buf, y: :f64_buf, n: :i64) do |k, g, w, x, y, n|
      accumulators = (0...DIMENSION).map { k.let(0.0) }
      k.count(n) do |i|
        base = k.let(i * DIMENSION)
        # 微分する式は素の式として組む。共有は Emitter が k.let に戻す。
        residual = prediction(w, x, base) - y[i]
        Nabla.emit_gradient(k, residual * residual, (0...DIMENSION).map { |j| w[j] })
             .each_with_index { |partial, j| k.assign(accumulators[j], accumulators[j] + partial) }
      end
      (0...DIMENSION).each { |j| k.store(g, j, accumulators[j]) }
      k.ret 0
    end
  end

  def prediction(weights, features, base)
    (0...DIMENSION).inject(Moissanite::Expr.lift(0.0, :f64)) do |sum, j|
      sum + (weights[j] * features[base + j])
    end
  end

  def fixture(samples)
    random = Random.new(19)
    weights = Moissanite::Buffer.f64(Array.new(DIMENSION) { random.rand - 0.5 })
    features = Moissanite::Buffer.f64(Array.new(samples * DIMENSION) { random.rand - 0.5 })
    labels = Moissanite::Buffer.f64(Array.new(samples) { random.rand - 0.5 })
    [weights, features, labels]
  end

  # dL/dw_j = sum_i 2 * (w.x_i - y_i) * x_ij
  def analytic_gradient(weights, features, labels, samples)
    gradient = Array.new(DIMENSION, 0.0)
    samples.times do |i|
      base = i * DIMENSION
      predicted = (0...DIMENSION).sum { |j| weights[j] * features[base + j] }
      residual = predicted - labels[i]
      (0...DIMENSION).each { |j| gradient[j] += 2 * residual * features[base + j] }
    end
    gradient
  end

  def test_gradient_through_a_sum_reduction_matches_the_analytic_solution
    samples = 2000
    weights, features, labels = fixture(samples)
    out = Moissanite::Buffer.f64(DIMENSION)
    gradient_kernel.call(out, weights, features, labels, samples)
    expected = analytic_gradient(weights, features, labels, samples)

    DIMENSION.times do |j|
      assert_in_delta expected[j], out[j], expected[j].abs * 1e-12,
                      "dL/dw#{j} must match the analytic least-squares gradient"
    end
  end

  def test_reduction_gradient_agrees_with_the_oracle_bit_for_bit
    samples = 500
    weights, features, labels = fixture(samples)
    native = Moissanite::Buffer.f64(DIMENSION)
    oracle = Moissanite::Buffer.f64(DIMENSION)
    kernel = gradient_kernel
    kernel.call(native, weights, features, labels, samples)
    kernel.interpret(oracle, weights, features, labels, samples)

    assert_equal oracle.to_a, native.to_a
  end

  # 勾配は全パラメータ分を 1 回の native 呼び出しで返す。損失そのものと
  # 同じ回数しかメモリを舐めない = 逆モードの効き目そのもの。
  def test_gradient_needs_one_pass_like_the_loss_itself
    samples = 1000
    weights, features, labels = fixture(samples)
    out = Moissanite::Buffer.f64(DIMENSION)

    assert_equal 0, gradient_kernel.call(out, weights, features, labels, samples)
    refute_equal Array.new(DIMENSION, 0.0), out.to_a,
                 'a silent zero gradient is the failure this library exists to prevent'
    assert_operator loss_kernel.call(weights, features, labels, samples), :>, 0.0
  end
end
