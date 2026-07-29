# frozen_string_literal: true

require_relative 'test_helper'

# 記号代数層 — 簡約・多項式・積分・重積分・極限。
#
# 積分のテストは値を手で書き写すのではなく、**微分し戻して検証する**。
# それがこの層の設計上の主張そのものなので、テストもその形を取る。
class SymbolicTest < Minitest::Test
  include NablaTest::Helpers

  def var(name = :x, index = 0)
    Moissanite::Param.new(type: :f64, name: name, index: index)
  end

  # ---- simplify ----------------------------------------------------------

  def test_identities_and_constant_folding
    x = var

    assert_equal x, Nabla.simplify((x * 1.0) + 0.0)
    assert_in_delta(0.0, Nabla.simplify(x * 0.0).value)
    assert_in_delta(7.0, Nabla.simplify((Moissanite::Expr.lift(3.0, :f64) * 2.0) + 1.0).value)
    assert_equal x, Nabla.simplify(x - 0.0)
    assert_in_delta(0.0, Nabla.simplify(x - x).value)
  end

  def test_simplify_shrinks_derivatives
    x = var
    raw = Nabla.derivative(x * x, x)
    simplified = Nabla.simplify(raw)

    assert_operator Nabla.size(simplified), :<, Nabla.size(raw)
    # 1.0*x + x*1.0  ->  2.0*x
    assert_equal %i[* const], [simplified.op, simplified.lhs.to_sexp.first]
    assert_in_delta 2.0, simplified.lhs.value, 0.0
  end

  # 0 除算は畳まない。moissanite では oracle が MathError を上げ native は
  # 未定義なので、簡約器が勝手に Inf を作ってはいけない。
  def test_division_by_zero_is_left_alone
    result = Nabla.simplify(Moissanite::Expr.lift(1.0, :f64) / Moissanite::Expr.lift(0.0, :f64))

    refute_instance_of Moissanite::Const, result
  end

  # 定義域の外も畳まない。
  def test_out_of_domain_math_is_left_alone
    refute_instance_of Moissanite::Const, Nabla.simplify(Moissanite::Expr.lift(-1.0, :f64).log)
    assert_in_delta Math.log(2.0), Nabla.simplify(Moissanite::Expr.lift(2.0, :f64).log).value, 1e-15
  end

  # ---- polynomial view ---------------------------------------------------

  def test_polynomial_view_folds_repeated_multiplication_into_degrees
    x = var

    assert_equal({ 3 => 1.0 }, Nabla::Poly.of(x * x * x, x))
    assert_equal({ 2 => 1.0, 1 => -1.0, 0 => -2.0 }, Nabla::Poly.of((x + 1.0) * (x - 2.0), x))
  end

  # 「多項式として見えなかった」を「0 次だった」に丸めない。
  def test_non_polynomial_returns_nil_rather_than_a_wrong_answer
    x = var

    assert_nil Nabla::Poly.of(x.sin, x)
    assert_nil Nabla::Poly.of(1.0 / (x + 1.0), x)
  end

  # ---- integration, verified by differentiating back ---------------------

  def integration_cases
    x = var
    {
      'polynomial' => (3.0 * x * x) + (2.0 * x) + 1.0,
      'expanded product' => (x + 1.0) * (x - 2.0),
      'sin of a linear argument' => ((2.0 * x) + 1.0).sin,
      'exp of a negated argument' => (-x).exp,
      'scaled cos' => 5.0 * (3.0 * x).cos,
      'reciprocal of a linear form' => 1.0 / ((2.0 * x) + 3.0),
      'constant' => Moissanite::Expr.lift(4.0, :f64)
    }
  end

  def test_every_antiderivative_survives_differentiating_it_back
    x = var
    integration_cases.each do |label, integrand|
      primitive = Nabla.integrate(integrand, x)
      result = Nabla.verify_integral(integrand, primitive, x, samples: 500)

      assert_predicate result, :ok?, "∫ #{label}: differentiating the answer back disagreed (#{result})"
    end
  end

  # 記号一致した積分は、微分規則が正しい限り正しい。数値は関与しない。
  # この件数が簡約器の強さの指標になる。
  def test_at_least_one_integral_verifies_symbolically
    x = var
    statuses = integration_cases.values.map do |integrand|
      Nabla.verify_integral(integrand, Nabla.integrate(integrand, x), x, samples: 50).status
    end

    assert_includes statuses, :symbolic,
                    'a symbolic match is a certificate; losing all of them means the simplifier regressed'
  end

  # 解けない形は近似を返さず断る。呼び手が成功と区別できない答えが
  # 一番危険なので、ここは「できない」と言えることをテストする。
  def test_unsupported_integrands_are_refused_rather_than_approximated
    x = var

    assert_raises(Nabla::NotElementary) { Nabla.integrate(x.sin * x.cos, x) } # needs parts
    assert_raises(Nabla::NotElementary) { Nabla.integrate(1.0 / ((x * x) + 1.0), x) } # needs arctan
    assert_raises(Nabla::NotElementary) { Nabla.integrate((x * x).sin, x) } # not elementary at all
  end

  # ---- definite and multiple integrals -----------------------------------

  def test_definite_integral_matches_the_exact_value
    x = var
    # ∫_0^2 (3x^2 + 2x + 1) dx = [x^3 + x^2 + x]_0^2 = 14
    assert_in_delta 14.0, Nabla.definite_integral((3.0 * x * x) + (2.0 * x) + 1.0, x, 0.0, 2.0).value, 1e-12
  end

  # 重積分に専用の機構は要らない: 内側が式として返るので外側に渡せる。
  def test_double_integral_is_just_nesting
    x = var
    y = var(:y, 1)

    inner = Nabla.definite_integral(x * y, x, 0.0, 2.0)          # = 2y
    outer = Nabla.definite_integral(inner, y, 0.0, 1.0)          # = 1

    assert_in_delta 1.0, outer.value, 1e-12

    squares = Nabla.definite_integral((x * x) + (y * y), x, 0.0, 1.0)
    total = Nabla.definite_integral(squares, y, 0.0, 1.0)

    assert_in_delta 2.0 / 3, total.value, 1e-12
  end

  def test_definite_integral_accepts_symbolic_bounds
    x = var
    y = var(:y, 1)
    # ∫_0^y 2x dx = y^2
    result = Nabla.definite_integral(2.0 * x, x, 0.0, y)

    assert_in_delta 9.0, Nabla.simplify(Nabla.substitute(result, y, Moissanite::Expr.lift(3.0, :f64))).value, 1e-12
  end

  # ---- limits ------------------------------------------------------------

  def test_lhopital_resolves_the_standard_indeterminate_forms
    x = var

    {
      'sin(x)/x' => [x.sin, x, 1.0],
      '(exp(x)-1)/x' => [x.exp - 1.0, x, 1.0],
      '(1-cos(x))/x^2' => [1.0 - x.cos, x * x, 0.5],
      'sin(3x)/sin(2x)' => [(3.0 * x).sin, (2.0 * x).sin, 1.5]
    }.each do |label, (numerator, denominator, expected)|
      assert_in_delta expected, Nabla.limit(numerator, denominator, x, at: 0.0), 1e-9, "lim #{label}"
    end
  end

  def test_a_limit_that_never_resolves_is_refused
    x = var

    assert_raises(Nabla::NotElementary) do
      Nabla.limit(x.sin, x.sin, x, at: 0.0, applications: 1)
    end
  end

  # ---- free_of? ----------------------------------------------------------

  # 依存を見落とすと定数として括り出され、答えが静かに間違う。
  def test_dependence_detection
    x = var
    y = var(:y, 1)

    assert Nabla.free_of?(y * 2.0, x)
    refute Nabla.free_of?((y * 2.0) + x.sin, x)
    refute Nabla.free_of?(x, x)
  end
end
