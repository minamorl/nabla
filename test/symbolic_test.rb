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

  def test_zero_integrand_rejects_a_wrong_primitive
    x = var
    result = Nabla.verify_integral(Moissanite::Expr.lift(0.0, :f64), x, x, samples: 10)

    refute_predicate result, :ok?
    assert_in_delta 1.0, result.max_relative_error, 0.0
  end

  def test_zero_expectation_uses_absolute_error_and_nonzero_uses_relative_error
    x = var
    zero = Moissanite::Expr.lift(0.0, :f64)
    small = Nabla.verify_integral(zero, 1e-10 * x, x, samples: 5)
    relative = Nabla.verify_integral(Moissanite::Expr.lift(2.0, :f64), x, x, samples: 5)

    assert_predicate small, :ok?
    assert_in_delta 1e-10, small.max_relative_error, 1e-20
    assert_in_delta 0.5, relative.max_relative_error, 0.0
    refute_predicate relative, :ok?
  end

  def test_entirely_undefined_integrand_is_inconclusive
    x = var
    result = Nabla.verify_integral(1.0 / (x - x), x, x, samples: 5)

    assert_equal :inconclusive, result.status
    refute_predicate result, :ok?
    assert_equal 'numeric comparison inconclusive', result.to_s
  end

  def test_substitution_preserves_an_isolated_singularity
    x = var
    result = Nabla.verify_integral(x / x, x + (x * x), x, samples: 5, range: 0.0..0.0)

    assert_equal :inconclusive, result.status
    refute_predicate result, :ok?
  end

  def test_primitive_singularity_is_not_hidden_by_simplification
    x = var
    result = Nabla.verify_integral((x + 1.0) - x, x + (x / x), x, samples: 5, range: 0.0..0.0)

    refute_predicate result, :ok?
  end

  def test_nonzero_fixed_point_preserves_singularities
    x = var
    d = x - 0.45
    primitive = Nabla.verify_integral((x + 1.0) - x, x + (d / d), x, samples: 1, range: 0.45..0.45)
    integrand = Nabla.verify_integral(d / d, x + (d * d), x, samples: 1, range: 0.45..0.45)

    refute_predicate primitive, :ok?
    assert_equal :inconclusive, integrand.status
  end

  def test_symbolic_additive_integration_constant
    x = var
    y = var(:y, 1)
    one = (x + 1.0) - x

    assert_predicate Nabla.verify_integral(one, x + y, x, samples: 1), :ok?
    assert_predicate Nabla.verify_integral(one - 1.0, y, x, samples: 1), :ok?
    assert_predicate Nabla.verify_integral(one, x + (2.0 * y), x, samples: 1), :ok?
    assert_predicate Nabla.verify_integral(one - 1.0, Moissanite::Expr.lift(7, :i64), x, samples: 1), :ok?
  end

  def test_nested_additive_integration_constants
    x = var
    y = var(:y, 1)
    one = (x + 1.0) - x
    cases = [[-(x + y), -one], [2.0 * (x + y), 2.0 * one],
             [(x + y) / 2.0, one / 2.0], [(x + y) * 1.0, one]]

    cases.each do |primitive, integrand|
      assert_predicate Nabla.verify_integral(integrand, primitive, x, samples: 3), :ok?
    end
  end

  def test_underflowed_polynomial_coefficients_do_not_prove_zero
    x = var
    small = 1e-200 * x
    squared = small * small
    primitive = -1.0 / squared
    derivative = Nabla.simplify(Nabla.derivative(primitive, x))

    assert_predicate Nabla.verify_integral(derivative, primitive, x, range: 1e200..1e200, samples: 3), :symbolic?
    assert_predicate Nabla.verify_integral(2.0 / x, squared.log, x, range: 1e200..1e200, samples: 3), :ok?
    one = x / 1e200

    assert_predicate Nabla.verify_integral(one, x + (0.0 * (1.0 / squared)), x, range: 1e200..1e200, samples: 3), :ok?
    rounded_zero = (x + 1.0) - x
    result = Nabla.verify_integral(rounded_zero, x + (0.0 * (1.0 / squared)), x, range: 1e200..1e200, samples: 3)

    assert_in_delta 1.0, result.max_relative_error, 0.0
    refute_predicate result, :ok?
  end

  def test_nonfinite_primitive_is_numeric_counterexample_at_finite_integrand
    x = var
    one = (x + 1.0) - x
    zero = (x.sin + 1.0) - (1.0 + x.sin)
    [x + zero.log, (2.0 * x) + Float::INFINITY].each do |primitive|
      result = Nabla.verify_integral(one, primitive, x, samples: 5)

      assert_equal :numeric, result.status
      assert_equal Float::INFINITY, result.max_relative_error
      refute_predicate result, :ok?
    end
  end

  def test_nonlinear_free_variable_is_inconclusive_in_numeric_fallback
    x = var
    y = var(:y, 1)
    one = (x + 1.0) - x
    [x + (1.0 / ((x + y) - x)), x + (0.0 * (x + y).log), x * (x + y)].each do |primitive|
      result = Nabla.verify_integral(one, primitive, x, samples: 3)

      assert_equal :inconclusive, result.status
      refute_predicate result, :ok?
    end
  end

  def test_unbound_integrand_is_inconclusive_before_numeric_compilation
    x = var
    y = var(:y, 1)
    zero = Moissanite::Expr.lift(0.0, :f64)
    cases = [[0.0 * (1.0 / y), zero], [y, x + Float::INFINITY], [1.0 / y, x + Float::INFINITY]]

    cases.each do |integrand, primitive|
      result = Nabla.verify_integral(integrand, primitive, x, samples: 5)

      assert_equal :inconclusive, result.status
      assert_equal 5, result.samples
      refute_predicate result, :ok?
    end
    assert_raises(ArgumentError) { Nabla.verify_integral(0.0 * (1.0 / y), zero, x, samples: 0) }
    assert_raises(ArgumentError) { Nabla.verify_integral(y, x + Float::INFINITY, x, range: 1.0...1.0) }
  end

  def test_certifiably_invalid_expressions_cannot_match_symbolically
    x = var
    one = (x + 1.0) - x
    zero = Moissanite::Expr.lift(0.0, :f64)
    bad = [x + (1.0 / zero), x + zero.log, x + Moissanite::Expr.lift(-1.0, :f64).log,
           x + (Moissanite::Expr.lift(1e308, :f64) + 1e308)]
    bad.each do |primitive|
      [Moissanite::Expr.lift(1.0, :f64), one].each do |integrand|
        refute_predicate Nabla.verify_integral(integrand, primitive, x, samples: 3), :ok?
      end
    end
  end

  def test_nested_static_select_ignores_unselected_infinity
    x = var
    cond = Moissanite::Expr.select(true, true, false)
    primitive = Moissanite::Expr.select(cond, x, x + Float::INFINITY)

    assert_predicate Nabla.verify_integral((x + 1.0) - x, primitive, x, samples: 1), :ok?
  end

  def test_invalid_ranges_and_excluded_upper_endpoint
    x = var
    one = (x + 1.0) - x

    [nil, (0.0..), (..1.0), (1.0...1.0), ((1 - Rational(1, 10**20))...1)].each do |range|
      assert_raises(ArgumentError) { Nabla.verify_integral(one, x, x, samples: 20, range: range) }
    end
    upper = 1.0.next_float
    result = Nabla.verify_integral(one, x + (1.0 / (x - upper)), x, samples: 100, range: 1.0...upper)

    assert_predicate result.max_relative_error, :finite?
  end

  def test_rewriters_preserve_shared_dag_identity
    x = var
    node = x
    20.times { node += node }
    resolved = Nabla::Verify.resolve_static_selects(node)
    stripped = Nabla::Verify.strip_additive_constants(resolved, x)

    assert_same stripped.lhs, stripped.rhs
  end

  def test_reachable_invalid_children_survive_shallow_folding
    x = var
    y = var(:y, 1)
    one = (x + 1.0) - x
    zero_np = (x.sin + 1.0) - (1.0 + x.sin)
    bad = [x + (1.0 / (one - 1.0)), x + (one - 1.0).log,
           x + ((x - x) - 1.0).sqrt, x + (0.0 * (y / 0.0)), x + zero_np.log,
           x + (1000.0 * (zero_np + 1.0)).exp,
           x + ((x - x) + 1e308 + 1e308)]

    bad.each do |primitive|
      result = Nabla.verify_integral(Moissanite::Expr.lift(1.0, :f64), primitive, x, samples: 8)

      refute_predicate result, :ok?
      refute_predicate result, :symbolic?
    end
  end

  def test_rational_ranges_use_exact_endpoint_comparisons
    x = var
    one = (x + 1.0) - x
    tiny = Rational(1, 10**400)
    ranges = [(tiny..(2 * tiny)), ((1 + Rational(1, 10**20))..(1 + Rational(2, 10**20))),
              ((1 - Rational(2, 10**20))..(1 - Rational(1, 10**20)))]

    ranges.each { |range| assert_raises(ArgumentError) { Nabla.verify_integral(one, x, x, range: range) } }
  end

  def test_shared_constant_dag_and_dynamic_select_condition
    x = var
    constant = Moissanite::Expr.lift(1.0, :f64)
    condition = x < 0.0
    20.times do
      constant += constant
      condition = Moissanite::Expr.select(condition, condition, condition)
    end
    resolved = Nabla::Verify.resolve_static_selects(condition)

    refute Nabla::Verify.definitely_invalid?(constant, x)
    assert_same resolved.cond, resolved.then_e
    assert_same resolved.cond, resolved.else_e
  end

  def test_dynamic_select_rejects_only_necessarily_invalid_cases
    x = var
    bad = Moissanite::Expr.lift(1.0, :f64) / 0.0
    condition = x < 0.0

    refute Nabla::Verify.definitely_invalid?(Moissanite::Expr.select(condition, bad, x), x)
    assert Nabla::Verify.definitely_invalid?(Moissanite::Expr.select(condition, bad, bad), x)
    assert Nabla::Verify.definitely_invalid?(Moissanite::Expr.select(bad < 0.0, x, x), x)
    selected = Nabla::Verify.resolve_static_selects(Moissanite::Expr.select(true, x, bad))

    refute Nabla::Verify.definitely_invalid?(selected, x)
  end

  def test_dynamic_select_cannot_be_erased_from_a_symbolic_certificate
    x = var
    one = Moissanite::Expr.lift(1.0, :f64)
    [Float::INFINITY, Float::NAN].each do |value|
      branch = Moissanite::Expr.select(x < 0.0, x + value, x)
      result = Nabla.verify_integral(one, x + (0.0 * branch), x, range: -2.0..-1.0, samples: 4)

      refute_predicate result, :symbolic?
      refute_predicate result, :ok?
    end
  end

  def test_overflowing_canceled_arithmetic_is_not_symbolic
    x = var
    one = Moissanite::Expr.lift(1.0, :f64)
    algebraic_one = (x + 1.0) - x
    nonpolynomial_one = ((x.sin + 1.0) - (1.0 + x.sin)) + 1.0
    big = 1e308 * algebraic_one
    other = 1e308 * nonpolynomial_one
    mid = 1e200 * algebraic_one
    [big + big, other + other, mid * mid, 0.0 * (big + big)].each do |expression|
      result = Nabla.verify_integral(one, x + expression, x, range: 0.4..1.6, samples: 4)

      refute_predicate result, :symbolic?
      refute_predicate result, :ok?
    end
  end

  def test_finite_constant_math_preserves_symbolic_without_samples
    x = var
    one = Moissanite::Expr.lift(1.0, :f64)
    two = Moissanite::Expr.lift(2.0, :f64)
    [two.log, two.sqrt, two.exp].each do |constant|
      result = Nabla.verify_integral(one, x + constant, x, samples: 0)

      assert_predicate result, :symbolic?
      assert_equal 0, result.samples
    end
  end

  def test_integrated_free_parameters_keep_symbolic_certificates
    x = var
    y = var(:y, 1)
    [y, 2.0 * y, y + 1.0].each do |integrand|
      result = Nabla.verify_integral(integrand, Nabla.integrate(integrand, x), x, samples: 0)

      assert_predicate result, :symbolic?
      assert_equal 0, result.samples
    end
    assert_predicate Nabla.verify_integral(Moissanite::Expr.lift(1.0, :f64), x + y, x, samples: 0), :symbolic?
    assert_predicate Nabla.verify_integral(Moissanite::Expr.lift(0.0, :f64), 2.0 * y, x, samples: 0), :symbolic?
  end

  def test_positive_exponential_bounds_preserve_symbolic_certificates
    x = var
    [[x.exp, x.exp, 0.31..3.7], [2.0 * x.exp, x.exp + x.exp, 0.31..3.7],
     [(-x).exp, -(-x).exp, -1.0..1.0]].each do |integrand, primitive, range|
      result = Nabla.verify_integral(integrand, primitive, x, range: range, samples: 0)

      assert_predicate result, :symbolic?
      assert_equal 0, result.samples
    end
  end

  def test_integrand_overflow_cannot_be_simplified_to_symbolic_zero
    x = var
    zero = Moissanite::Expr.lift(0.0, :f64)
    one = (x + 1.0) - x
    huge = (1e308 * one) + (1e308 * one)
    result = Nabla.verify_integral(huge - huge, zero, x, samples: 5)

    assert_equal :inconclusive, result.status
    refute_predicate result, :ok?
  end

  def test_primitive_division_by_zero_is_not_accepted
    x = var
    one = (x + 1.0) - x
    result = Nabla.verify_integral(one, x + (1.0 / (one - 1.0)), x, range: 1.0..1.0, samples: 1)

    refute_predicate result, :ok?
  end

  def test_unselected_nonfinite_constant_is_ignored
    x = var
    one = (x + 1.0) - x
    primitive = Moissanite::Expr.select(true, x, x + Float::INFINITY)

    assert_predicate Nabla.verify_integral(one, primitive, x, samples: 1), :ok?
  end

  def test_empty_exclusive_range_is_rejected
    x = var
    one = (x + 1.0) - x

    assert_raises(ArgumentError) { Nabla.verify_integral(one, x, x, range: 1.0...1.0, samples: 1) }
  end

  def test_explicit_nonfinite_primitive_is_never_accepted
    x = var
    [Float::INFINITY, Float::NAN].each do |value|
      [Moissanite::Expr.lift(1.0, :f64), (x + 1.0) - x].each do |integrand|
        result = Nabla.verify_integral(integrand, x + value, x, samples: 1)

        refute_predicate result, :ok?
      end
    end
  end

  def test_nonfinite_primitive_math_does_not_raise
    x = var
    nan = Moissanite::Expr.lift(Float::NAN, :f64)
    result = Nabla.verify_integral((x + 1.0) - x, x * nan.min(nan), x, samples: 1)

    refute_predicate result, :ok?
  end

  def test_large_finite_range_does_not_sample_infinity
    x = var
    result = Nabla.verify_integral(x.min(1.0), x, x, samples: 50, range: -Float::MAX..Float::MAX)

    assert_equal :numeric, result.status
    refute_predicate result, :ok?
  end

  def test_nan_constant_comparison_is_inconclusive
    x = var
    nan = Moissanite::Expr.lift(Float::NAN, :f64)
    result = Nabla.verify_integral(nan.min(nan), x, x, samples: 5)

    assert_equal :inconclusive, result.status
    refute_predicate result, :ok?
  end

  def test_simplified_derivative_preserves_existing_numeric_callers
    x = var
    integrand = (x + 1.0) - x
    small = 1e-200 * x
    selected = Moissanite::Expr.select(true, x + (small / small), x)
    [x + (0.0 * x.abs), x + (small / small), selected].each do |primitive|
      result = Nabla.verify_integral(integrand, primitive, x, samples: 1, range: 1.0..1.0)

      assert_predicate result, :ok?
    end
  end

  def test_nonfinite_derivative_at_finite_expectation_is_a_counterexample
    x = var
    result = Nabla.verify_integral(Moissanite::Expr.lift(1.0, :f64), x.log, x, samples: 5, range: 0.0..0.0)

    assert_equal :numeric, result.status
    assert_equal Float::INFINITY, result.max_relative_error
    refute_predicate result, :ok?
    refute result.ok?(Float::INFINITY)
  end

  def test_relative_error_avoids_overflow_for_finite_opposite_values
    x = var
    expected = Moissanite::Expr.lift(Float::MAX, :f64)
    result = Nabla.verify_integral(expected, -Float::MAX * x, x, samples: 5, range: 0.31..0.9)

    assert_in_delta 2.0, result.max_relative_error, 0.0
    assert result.ok?(2.1)
  end

  def test_domain_error_in_derivative_is_a_counterexample
    x = var
    result = Nabla.verify_integral(Moissanite::Expr.lift(1.0, :f64), x.log.log, x, samples: 5, range: -1.0..-1.0)

    assert_equal :numeric, result.status
    assert_equal Float::INFINITY, result.max_relative_error
    refute_predicate result, :ok?
  end

  def test_symbolic_certificate_requires_the_original_domain_guards
    x = var
    one = Moissanite::Expr.lift(1.0, :f64)
    integrand = one / (2.0 * x.sqrt)
    primitives = [x.sqrt + (0.0 * (x - 2.0).log), x.sqrt + (0.0 * (x - 2.0).sqrt)]

    primitives.each do |primitive|
      result = Nabla.verify_integral(integrand, primitive, x, samples: 4, range: 0.4..1.6)

      assert_equal :numeric, result.status
      assert_equal Float::INFINITY, result.max_relative_error
      refute_predicate result, :ok?
    end
  end

  def test_preserved_sqrt_guard_remains_symbolic
    x = var
    one = Moissanite::Expr.lift(1.0, :f64)
    result = Nabla.verify_integral(one / (2.0 * x.sqrt), x.sqrt, x, samples: 4, range: 0.4..1.6)

    assert_predicate result, :symbolic?
  end

  def test_unproven_log_guard_uses_numeric_agreement
    x = var
    result = Nabla.verify_integral(1.0 / x, x.log, x, samples: 10, range: 0.4..1.6)

    assert_equal :numeric, result.status
    assert_predicate result, :ok?
  end

  def test_domain_error_in_integrand_is_inconclusive
    x = var
    result = Nabla.verify_integral(x.log, x, x, samples: 5, range: -1.0..-1.0)

    assert_equal :inconclusive, result.status
    refute_predicate result, :ok?
  end

  def test_sampling_count_is_validated_only_for_numeric_comparison
    x = var
    zero = Moissanite::Expr.lift(0.0, :f64)

    [0, -1, 1.5, nil].each do |count|
      assert_raises(ArgumentError) { Nabla.verify_integral(zero, x, x, samples: count) }
    end

    assert_predicate Nabla.verify_integral(Moissanite::Expr.lift(1.0, :f64), x, x, samples: 0), :symbolic?
  end

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
