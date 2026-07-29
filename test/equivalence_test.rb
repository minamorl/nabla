# frozen_string_literal: true

require_relative 'test_helper'

# moissanite の憲法をそのまま継ぐ検証。
#
#   法則2「oracle が意味論の正典」— 導関数カーネルも例外ではない。
#   純 Ruby の oracle と native backend が**ビット一致**しなければ、
#   その導関数は信用できない。
#
# 加えて、前進モードと逆モードの相互検証をここに置く。両者は独立実装で
# 同じ値を出すはずなので、片方のバグをもう片方が捕まえる。
class EquivalenceTest < Minitest::Test
  include NablaTest::Helpers

  def buffer_param
    Moissanite::Param.new(type: :f64_buf, name: :xs, index: 0)
  end

  def sample(buf, count)
    squares = (0...count).inject(Moissanite::Expr.lift(1e-6, :f64)) do |acc, i|
      d = buf[i] - (0.5 + (i * 0.01))
      acc + (d * d)
    end
    squares.log * (1.0 + buf[0].sin)
  end

  def reverse_kernel(count)
    Moissanite.kernel(:"rev#{count}", o: :f64_buf, buf: :f64_buf, n: :i64) do |k, o, buf, _n|
      Nabla.emit_gradient(k, sample(buf, count), (0...count).map { |i| buf[i] })
           .each_with_index { |partial, i| k.store(o, i, partial) }
      k.ret 0
    end
  end

  def forward_kernel(count)
    Moissanite.kernel(:"fwd#{count}", o: :f64_buf, buf: :f64_buf, n: :i64) do |k, o, buf, _n|
      root = sample(buf, count)
      (0...count).each { |i| k.store(o, i, Nabla.derivative(root, buf[i])) }
      k.ret 0
    end
  end

  def test_derivative_kernel_agrees_with_the_oracle_bit_for_bit
    kernel = reverse_kernel(6)
    point = Moissanite::Buffer.f64([0.9, 0.3, 1.4, -0.2, 2.7, -1.1])
    native = Moissanite::Buffer.f64(6)
    oracle = Moissanite::Buffer.f64(6)
    kernel.call(native, point, 6)
    kernel.interpret(oracle, point, 6)

    assert_equal oracle.to_a, native.to_a,
                 'the derivative kernel must be indistinguishable from the oracle, like any other kernel'
  end

  def test_high_order_derivative_also_agrees_with_the_oracle
    x = Moissanite::Param.new(type: :f64, name: :x, index: 0)
    third = Moissanite.kernel(:third, x: :f64) do |k, param|
      potential = -0.5 * (((param - 100.0) * (param - 100.0)) + 1e-6).log
      d3 = 3.times.inject(potential) { |acc, _| Nabla.derivative(acc, param) }
      k.ret Nabla.emit(k, d3)
    end

    [100.5, 103.25, 97.75].each do |value|
      assert_equal third.interpret(value), third.call(value),
                   "3rd derivative must be bit-identical between oracle and native at x=#{value}"
    end
    refute_nil x
  end

  # 前進と逆は随伴の加算順序が違うので、浮動小数点として最終 ulp が
  # ずれうる。**それは欠陥ではない** — だから厳密一致ではなく、
  # 「数 ulp 以内」で縛る。各モードが自分の oracle と一致することは
  # 上のテストが別途保証している。
  def test_forward_and_reverse_agree_to_within_a_few_ulp
    count = 5
    point = Moissanite::Buffer.f64([0.9, 0.3, 1.4, -0.2, 2.7])
    from_reverse = Moissanite::Buffer.f64(count)
    from_forward = Moissanite::Buffer.f64(count)
    reverse_kernel(count).call(from_reverse, point, count)
    forward_kernel(count).call(from_forward, point, count)

    count.times do |i|
      expected = from_forward[i]
      actual = from_reverse[i]
      tolerance = [expected.abs, 1.0].max * Float::EPSILON * 4

      assert_in_delta expected, actual, tolerance,
                      "partial #{i}: modes may differ by association order, but only in the last ulps"
    end
  end
end
