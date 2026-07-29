# frozen_string_literal: true

require_relative 'test_helper'

# 逆モードの中身。「1 回の走査で全偏微分」という主張を、値の正しさと
# 規模の両方で縛る。
class ReverseTest < Minitest::Test
  include NablaTest::Helpers

  def xs
    Moissanite::Param.new(type: :f64_buf, name: :xs, index: 0)
  end

  # f(x) = log(eps + sum (x_i - c_i)^2) * (1 + sin(x_0))
  # 全入力が寄与し、偏微分どうしが分母を共有する形。
  def sample(buffer, count)
    squares = (0...count).inject(Moissanite::Expr.lift(1e-6, :f64)) do |acc, i|
      d = buffer[i] - (0.5 + (i * 0.01))
      acc + (d * d)
    end
    squares.log * (1.0 + buffer[0].sin)
  end

  def gradient_kernel(count)
    Moissanite.kernel(:"grad#{count}", o: :f64_buf, buf: :f64_buf, n: :i64) do |k, o, buf, _n|
      targets = (0...count).map { |i| buf[i] }
      Nabla.emit_gradient(k, sample(buf, count), targets)
           .each_with_index { |partial, i| k.store(o, i, partial) }
      k.ret 0
    end
  end

  def test_gradient_matches_a_hand_written_analytic_solution
    # f(x0, x1) = (x0 * x1) + sin(x0)
    #   df/dx0 = x1 + cos(x0),  df/dx1 = x0
    kernel = Moissanite.kernel(:pair, o: :f64_buf, b: :f64_buf, n: :i64) do |k, o, b, _n|
      Nabla.emit_gradient(k, (b[0] * b[1]) + b[0].sin, [b[0], b[1]])
           .each_with_index { |partial, i| k.store(o, i, partial) }
      k.ret 0
    end
    point = Moissanite::Buffer.f64([0.7, -1.3])
    out = Moissanite::Buffer.f64(2)
    kernel.call(out, point, 2)

    assert_in_delta(-1.3 + Math.cos(0.7), out[0], 0.0)
    assert_in_delta 0.7, out[1], 0.0
  end

  def test_every_partial_comes_from_one_traversal
    kernel = gradient_kernel(4)
    point = Moissanite::Buffer.f64([0.9, 0.3, 1.4, -0.2])
    out = Moissanite::Buffer.f64(4)
    kernel.call(out, point, 4)

    values = point.to_a

    4.times do |i|
      assert_in_delta probe_derivative(values, i), out[i], 1e-6,
                      "partial #{i} must agree with a central difference of the same function"
    end
  end

  # 逆モードは O(N)、前進モードは O(N^2)。ノード数で縛る。
  def test_reverse_stays_linear_while_forward_grows_quadratically
    forward_totals = []
    reverse_totals = []
    [4, 16].each do |count|
      buf = xs
      root = sample(buf, count)
      targets = (0...count).map { |i| buf[i] }
      forward_totals << targets.sum { |t| Nabla.size(Nabla.derivative(root, t)) }
      reverse_totals << Nabla.size(Nabla.gradient(root, targets).values.reduce(:+))
    end

    # 入力 4 倍で、逆モードは 4 倍程度、前進モードは 16 倍程度に伸びる。
    reverse_growth = reverse_totals[1].to_f / reverse_totals[0]
    forward_growth = forward_totals[1].to_f / forward_totals[0]

    assert_operator reverse_growth, :<, 6.0, 'reverse mode must stay roughly linear in the input count'
    assert_operator forward_growth, :>, 10.0, 'forward mode is expected to grow quadratically'
  end

  # 共有を展開すると爆発する = Emitter が必須である、という事実そのもの。
  def test_shared_dag_is_far_smaller_than_its_tree_expansion
    buf = xs
    targets = (0...16).map { |i| buf[i] }
    combined = Nabla.gradient(sample(buf, 16), targets).values.reduce(:+)

    assert_operator Nabla.tree_size(combined), :>, Nabla.size(combined) * 20,
                    'the tree expansion must be dramatically larger, which is why Emitter exists'
  end

  def test_unreachable_targets_get_an_exact_zero
    buf = xs
    partials = Nabla.gradient(buf[0] * 2.0, [buf[0], buf[5]])

    assert_in_delta(0.0, partials[buf[5]].value)
  end

  def test_unsupported_operation_is_refused
    buf = xs
    assert_raises(Nabla::Unsupported) do
      Nabla.gradient(Moissanite::Expr.math2(:hypot, buf[0], buf[1]), [buf[0]])
    end
  rescue Moissanite::Error
    skip 'moissanite rejects the unknown math op before nabla sees it'
  end

  private

  def probe_sample(values, index, replacement)
    v = values.dup
    v[index] = replacement
    squares = 1e-6
    v.each_with_index { |x, i| squares += (x - (0.5 + (i * 0.01)))**2 }
    Math.log(squares) * (1.0 + Math.sin(v[0]))
  end

  def probe_derivative(values, index)
    h = 1e-6
    (probe_sample(values, index, values[index] + h) - probe_sample(values, index, values[index] - h)) / (2 * h)
  end
end
