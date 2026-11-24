# frozen_string_literal: true

module Nabla
  # ==================================================================
  # Verify — 積分の答えを微分して問いに戻れるか確かめる。
  #
  # **この層の存在理由**である。表引きの CAS は答えを返して終わりで、
  # 正しさは実装への信頼に帰着する。ここには微分器があるので、
  # ∫f = F の主張はそのまま F' = f という**検査可能な命題**になる。
  #
  # 二段構えで、二段目に落ちたことを隠さない:
  #
  #   :symbolic — F' を簡約したら f と構文的に一致した。微分規則が
  #               正しい限りこの積分は正しい。数値は一切関与しない。
  #               証明書と呼べるのはこの段だけである。
  #   :numeric  — 一致しなかったので、両者を native カーネルにして
  #               乱択した点で突き合わせた。反例が無いことしか言えない。
  #
  # 記号一致率は簡約器の強さの関数なので、そのまま開発の指標になる。
  # ==================================================================
  module Verify # rubocop:disable Metrics/ModuleLength
    Result = Data.define(:status, :max_relative_error, :samples) do
      def symbolic? = status == :symbolic

      def ok?(tolerance = 1e-9)
        symbolic? || (status == :numeric && max_relative_error.finite? && max_relative_error <= tolerance)
      end

      def to_s
        return 'symbolically exact' if symbolic?
        return 'numeric comparison inconclusive' if status == :inconclusive

        format('numeric comparison, max scaled error %.2e', max_relative_error)
      end
    end

    module_function

    # 既定の標本域は 0 と負数を避ける (log や 1/x の定義域の外で
    # 突き合わせても、積分が間違っている証拠にはならないため)。
    def integral(integrand, primitive, variable, samples: 2000, range: 0.31..3.7, seed: 11)
      validate_range(range)
      primitive = resolve_static_selects(primitive)
      integrand = resolve_static_selects(integrand)
      evidence = {}.compare_by_identity
      return inconclusive if fold_evidence(integrand, evidence)[1]

      invalid_primitive = fold_evidence(primitive, evidence)[1]
      differentiated = simplified_derivative(primitive, variable)
      if symbolic_certificate?(invalid_primitive, differentiated, integrand, primitive, variable, range, evidence)
        return Result.new(status: :symbolic, max_relative_error: 0.0, samples: 0)
      end

      return inconclusive if !invalid_primitive && !differentiated

      sampling = { samples: samples, range: range, seed: seed, invalid_primitive: invalid_primitive }
      numeric_result(integrand, differentiated, primitive, variable, sampling)
    end

    def numeric_result(integrand, differentiated, primitive, variable, sampling)
      samples = sampling.fetch(:samples)
      raise ArgumentError, 'samples must be a positive integer' unless samples.is_a?(Integer) && samples.positive?

      worst, comparisons = compare_numerically(integrand, differentiated, primitive, variable, sampling)
      Result.new(status: comparisons.zero? ? :inconclusive : :numeric, max_relative_error: worst, samples: samples)
    end

    def symbolic_certificate?(invalid, differentiated, integrand, primitive, variable, range, evidence) # rubocop:disable Metrics/ParameterLists
      !invalid && differentiated && symbolic_match?(differentiated, integrand) &&
        !lost_domain_checks?(primitive, differentiated, evidence) &&
        finite_over_range?(primitive, variable, range, evidence) &&
        finite_over_range?(integrand, variable, range, evidence)
    end

    def inconclusive
      Result.new(status: :inconclusive, max_relative_error: 0.0, samples: 0)
    end

    def strip_additive_constants(node, variable, memo = {}.compare_by_identity, additive: true)
      key = memo[node] ||= {}
      return key[additive] if key.key?(additive)

      dependencies = node_dependencies(node, variable, memo)
      if additive && dependencies == [false, true] && linear_constant?(node, memo)
        return key[additive] = Simplify.number(0.0)
      end

      key[additive] = strip_linear_node(node, variable, memo, additive)
    end

    def strip_linear_node(node, variable, memo, additive) # rubocop:disable Metrics/CyclomaticComplexity
      return node unless additive
      return node.with(expr: strip_additive_constants(node.expr, variable, memo)) if node.is_a?(Moissanite::Neg)
      return node unless node.is_a?(Moissanite::BinOp)

      left, right = linear_children(node)
      return node unless left || right

      node.with(lhs: left ? strip_additive_constants(node.lhs, variable, memo) : node.lhs,
                rhs: right ? strip_additive_constants(node.rhs, variable, memo) : node.rhs)
    end

    def finite_scalar?(node)
      node.is_a?(Moissanite::Const) && node.value.is_a?(Numeric) && node.value.finite?
    end

    def linear_children(node)
      return [true, true] if %i[+ -].include?(node.op)
      return [finite_scalar?(node.rhs), finite_scalar?(node.lhs)] if node.op == :*
      return [finite_scalar?(node.rhs) && !node.rhs.value.zero?, false] if node.op == :/

      [false, false]
    end

    def linear_constant?(node, memo)
      entry = memo[node] ||= {}
      return entry[:linear] if entry.key?(:linear)

      entry[:linear] = linear_constant_node?(node, memo)
    end

    def linear_constant_node?(node, memo) # rubocop:disable Metrics/CyclomaticComplexity, Metrics/PerceivedComplexity
      return true if node.is_a?(Moissanite::Param) || finite_scalar?(node)
      return linear_constant?(node.expr, memo) if node.is_a?(Moissanite::Neg)
      return false unless node.is_a?(Moissanite::BinOp)

      left, right = linear_children(node)
      (left || right) && (!left || linear_constant?(node.lhs, memo)) && (!right || linear_constant?(node.rhs, memo))
    end

    def node_dependencies(node, variable, memo)
      entry = memo[node] ||= {}
      return entry[:dependencies] if entry.key?(:dependencies)

      target = Rewrite.same?(node, variable)
      other = node.is_a?(Moissanite::Param) && !target
      DAG.children(node).each do |child|
        child_target, child_other = node_dependencies(child, variable, memo)
        target ||= child_target
        other ||= child_other
      end
      entry[:dependencies] = [target, other]
    end

    def resolve_static_selects(node, memo = {}.compare_by_identity)
      return memo[node] if memo.key?(node)

      original = node
      if node.is_a?(Moissanite::Select)
        cond = resolve_static_selects(node.cond, memo)
        if cond.is_a?(Moissanite::Const)
          return memo[node] = resolve_static_selects(cond.value ? node.then_e : node.else_e, memo)
        end

        return memo[original] = node.with(cond: cond, then_e: resolve_static_selects(node.then_e, memo),
                                          else_e: resolve_static_selects(node.else_e, memo))
      end

      replace = ->(child) { resolve_static_selects(child, memo) }
      memo[original] = rebuild_for_sampling(node, replace)
    end

    def definitely_invalid?(expression, variable)
      _ = variable
      fold_evidence(expression, {}.compare_by_identity)[1]
    end

    def fold_evidence(node, memo)
      return memo[node] if memo.key?(node)

      children = DAG.children(node).map { |child| fold_evidence(child, memo) }
      invalid = invalid_folded_children?(node, children)
      folded = shallow_fold(node, children.map(&:first))
      memo[node] = [folded, invalid || invalid_shallow?(folded)]
    rescue ArgumentError, Math::DomainError, FloatDomainError, ZeroDivisionError
      memo[node] = [node, true]
    end

    def invalid_folded_children?(node, children)
      return children.any? { |child| child[1] } unless node.is_a?(Moissanite::Select)

      children[0][1] || (children[1][1] && children[2][1])
    end

    def shallow_fold(node, children)
      case node
      when Moissanite::BinOp
        return node.with(lhs: children[0], rhs: children[1]) unless %i[+ - * /].include?(node.op)

        Simplify.binop(node.op, *children)
      when Moissanite::Neg then Simplify.negate(children[0])
      when Moissanite::MathOp then Simplify.math(node.fn, children)
      else
        index = 0
        rebuild_for_sampling(node, ->(_child) { children[index].tap { index += 1 } })
      end
    end

    def invalid_shallow?(node) # rubocop:disable Metrics/CyclomaticComplexity, Metrics/PerceivedComplexity
      return !node.value.finite? if node.is_a?(Moissanite::Const) && node.value.is_a?(Numeric)
      return node.rhs.value.zero? if node.is_a?(Moissanite::BinOp) && node.op == :/ && finite_scalar?(node.rhs)
      return false unless node.is_a?(Moissanite::MathOp) && %i[log sqrt].include?(node.fn)

      argument = node.args.first
      return false unless finite_scalar?(argument)

      node.fn == :log ? argument.value <= 0 : argument.value.negative?
    end

    def lost_domain_checks?(primitive, differentiated, evidence = {}.compare_by_identity)
      ids = {}.compare_by_identity
      intern = {}
      fold_evidence(primitive, evidence)
      fold_evidence(differentiated, evidence)
      required, conditional = domain_guards(primitive, ids, intern, evidence, factors: false)
      return true if conditional
      return false if required.empty?

      available, conditional = domain_guards(differentiated, ids, intern, evidence, factors: true)
      conditional || required.each_key.any? { |guard| !available.key?(guard) }
    end

    # A symbolic certificate may not erase an overflowing original operation.
    # Each arithmetic bound is rounded outwards to representable f64 values.
    def finite_over_range?(expression, variable, range, evidence)
      lower, upper = representable_bounds(range)
      context = { variable: variable, range: [lower.to_r, upper.to_r], evidence: evidence,
                  bounds: {}.compare_by_identity, dependencies: {}.compare_by_identity }
      !finite_bounds(expression, context).nil?
    end

    def finite_bounds(node, context)
      memo = context.fetch(:bounds)
      return memo[node] if memo.key?(node)

      memo[node] = compute_finite_bounds(node, context)
    end

    def compute_finite_bounds(node, context) # rubocop:disable Metrics/CyclomaticComplexity
      case node
      when Moissanite::Const
        return [node.value.to_r, node.value.to_r] if finite_scalar?(node)
      when Moissanite::Param
        return context.fetch(:range) if Rewrite.same?(node, context.fetch(:variable))
        return [0.to_r, 0.to_r] if %i[f64 i64].include?(node.type)
      when Moissanite::Neg
        child = finite_bounds(node.expr, context)
        return [-child[1], -child[0]] if child
      when Moissanite::BinOp
        return arithmetic_bounds(node, context)
      when Moissanite::MathOp
        return math_bounds(node, context)
      end
      nil
    end

    def arithmetic_bounds(node, context) # rubocop:disable Metrics/CyclomaticComplexity, Metrics/PerceivedComplexity
      left = finite_bounds(node.lhs, context)
      right = finite_bounds(node.rhs, context)
      return nil unless left && right

      values = case node.op
               when :+ then [left[0] + right[0], left[1] + right[1]]
               when :- then [left[0] - right[1], left[1] - right[0]]
               when :* then left.product(right).map { |a, b| a * b }.minmax
               when :/
                 return nil if 0.between?(right[0], right[1])

                 left.product(right).map { |a, b| a / b }.minmax
               end
      values && rounded_bounds(*values)
    end

    def rounded_bounds(lower, upper)
      limit = Float::MAX.to_r
      return nil if lower < -limit || upper > limit

      lo = lower.to_f
      hi = upper.to_f
      lo = lo.prev_float if lo.to_r > lower
      hi = hi.next_float if hi.to_r < upper
      return nil unless lo.finite? && hi.finite?

      [lo.to_r, hi.to_r]
    end

    def math_bounds(node, context)
      folded, invalid = context.fetch(:evidence).fetch(node)
      dependencies = node_dependencies(node, context.fetch(:variable), context.fetch(:dependencies))
      if !invalid && dependencies == [false, false] && finite_scalar?(folded)
        return [folded.value.to_r, folded.value.to_r]
      end

      bounds = node.args.map { |arg| finite_bounds(arg, context) }
      return nil if bounds.any?(&:nil?)

      simple_math_bounds(node.fn, bounds)
    end

    def simple_math_bounds(fn, bounds) # rubocop:disable Metrics/CyclomaticComplexity
      low, high = bounds.first
      limit = Float::MAX.to_r
      case fn
      when :sin, :cos then [-1.to_r, 1.to_r]
      when :sqrt then [[1.to_r, low].min, [1.to_r, high].max] if low >= 0
      when :log then [-limit, limit] if low.positive?
      when :exp then exp_bounds(high)
      when :abs then [0.to_r, [low.abs, high.abs].max]
      when :min, :max then [[low, bounds[1][0]].min, [high, bounds[1][1]].max]
      end
    end

    def exp_bounds(high)
      return [0.to_r, 1.to_r] if high <= 0
      return nil if high > 710

      # e < 8/3 + 5/96 = 87/32, bounding the factorial tail from n=4.
      rounded_bounds(0.to_r, Rational(87, 32)**high.ceil)
    end

    def domain_guards(root, ids, intern, evidence, factors:)
      seen = {}.compare_by_identity
      stack = [root]
      guards = {}
      conditional = false
      until stack.empty?
        node = stack.pop
        next if seen.key?(node)

        seen[node] = true
        conditional ||= node.is_a?(Moissanite::Select)
        record_guard(node, guards, ids, intern, evidence, factors)
        stack.concat(DAG.children(node))
      end
      [guards, conditional]
    end

    def record_guard(node, guards, ids, intern, evidence, factors) # rubocop:disable Metrics/ParameterLists
      if node.is_a?(Moissanite::BinOp) && node.op == :/
        division_guards(node.rhs, guards, ids, intern, factors)
      elsif node.is_a?(Moissanite::MathOp) && %i[log sqrt exp].include?(node.fn)
        folded, invalid = evidence.fetch(node)
        guards[[node.fn, expression_id(node.args.first, ids, intern)]] = true unless !invalid && finite_scalar?(folded)
      end
    end

    def division_guards(denominator, guards, ids, intern, factors)
      pending = [denominator]
      seen = {}.compare_by_identity
      until pending.empty?
        node = pending.pop
        next if seen.key?(node)

        seen[node] = true
        guards[[:division, expression_id(node, ids, intern)]] = true unless nonzero_scalar?(node)
        pending.push(node.lhs, node.rhs) if factors && product?(node)
      end
    end

    def nonzero_scalar?(node) = finite_scalar?(node) && !node.value.zero?
    def product?(node) = node.is_a?(Moissanite::BinOp) && node.op == :*

    # Intern small keys containing child IDs, never recursively hash expression DAGs.
    def expression_id(node, ids, intern)
      return ids[node] if ids.key?(node)

      children = DAG.children(node).map { |child| expression_id(child, ids, intern) }
      key = [node.class, *expression_attributes(node), *children]
      ids[node] = intern[key] ||= intern.size
    end

    def expression_attributes(node)
      case node
      when Moissanite::Const then [node.type, node.value]
      when Moissanite::Param then [node.type, node.name, node.index]
      when Moissanite::BinOp then [node.type, node.op]
      when Moissanite::MathOp then [node.type, node.fn]
      when Moissanite::Neg, Moissanite::Cast, Moissanite::Not, Moissanite::Select then [node.type]
      else [node.object_id]
      end
    end

    def derivative(primitive, variable)
      Forward.derivative(primitive, variable)
    end

    def simplified_derivative(primitive, variable)
      Simplify.run(derivative(primitive, variable))
    rescue ArgumentError, Math::DomainError, FloatDomainError, ZeroDivisionError
      nil
    end

    def symbolic_match?(differentiated, integrand)
      differentiated == Simplify.run(integrand)
    rescue ArgumentError
      false
    end

    def compare_numerically(integrand, differentiated, primitive, variable, sampling) # rubocop:disable Metrics/MethodLength, Metrics/CyclomaticComplexity, Metrics/PerceivedComplexity
      return [0.0, 0] if node_dependencies(integrand, variable, {}.compare_by_identity)[1]

      wanted = compile(integrand, variable)
      got, original = numeric_primitive_kernels(differentiated, primitive, variable, sampling)
      return [0.0, 0] if !sampling.fetch(:invalid_primitive) && !original

      random = Random.new(sampling.fetch(:seed))
      range = sampling.fetch(:range)
      lower, upper = representable_bounds(range)
      worst = 0.0
      comparisons = 0
      sampling.fetch(:samples).times do
        fraction = random.rand
        point = lower == upper ? lower : (lower * (1.0 - fraction)) + (upper * fraction)
        point = point.clamp(lower, upper)
        expected = evaluate(wanted, point)
        next unless expected.finite?

        comparisons += 1
        return [Float::INFINITY, comparisons] if sampling.fetch(:invalid_primitive)
        return [Float::INFINITY, comparisons] unless evaluate(original, point).finite?

        actual = evaluate(got, point)
        return [Float::INFINITY, comparisons] unless actual.finite?

        worst = [worst, scaled_error(actual, expected)].max
      end
      [worst, comparisons]
    end

    def numeric_primitive_kernels(differentiated, primitive, variable, sampling)
      return [nil, nil] if sampling.fetch(:invalid_primitive)

      checked = strip_additive_constants(primitive, variable)
      return [nil, nil] if node_dependencies(checked, variable, {}.compare_by_identity)[1]

      [compile(differentiated, variable, simplify: true), compile(checked, variable)]
    end

    def validate_range(range) # rubocop:disable Metrics/CyclomaticComplexity
      valid_endpoints = range.is_a?(Range) && [range.begin, range.end].all? { |v| finite_real?(v) }
      if valid_endpoints && (range.begin.to_r < range.end.to_r ||
                             (range.begin.to_r == range.end.to_r && !range.exclude_end?))
        lower, upper = representable_bounds(range)
        return if lower <= upper
      end

      raise ArgumentError, 'range must have finite, ordered endpoints'
    end

    def representable_bounds(range)
      lower = range.begin.to_f
      upper = range.end.to_f
      lower = lower.next_float if lower.to_r < range.begin.to_r
      upper = upper.prev_float if upper.to_r > range.end.to_r || (range.exclude_end? && upper.to_r >= range.end.to_r)
      [lower, upper]
    end

    def finite_real?(value)
      value.is_a?(Numeric) && !value.is_a?(Complex) && value.finite? && value.to_f.finite?
    end

    def scaled_error(actual, expected)
      scale = expected.zero? ? 1.0 : expected.abs
      return (actual.abs / scale) + (expected.abs / scale) if actual.negative? != expected.negative?

      ((actual - expected) / scale).abs
    end

    def evaluate(kernel, point)
      kernel.call(point)
    rescue Math::DomainError, ZeroDivisionError, FloatDomainError
      Float::NAN
    end

    def compile(expression, variable, simplify: false)
      Moissanite.kernel(:verify_point, x: :f64) do |builder, x|
        substituted = if simplify
                        Rewrite.substitute(expression, variable, x)
                      else
                        substitute_for_sampling(expression, variable, x)
                      end
        builder.ret contains_select?(substituted) ? substituted : Nabla.emit(builder, substituted)
      end
    end

    def contains_select?(root)
      seen = {}.compare_by_identity
      stack = [root]
      until stack.empty?
        node = stack.pop
        next if seen.key?(node)

        return true if node.is_a?(Moissanite::Select)

        seen[node] = true
        stack.concat(DAG.children(node))
      end
      false
    end

    # Numeric evaluation must preserve singularities rather than simplify x/x to 1.
    def substitute_for_sampling(node, variable, replacement, memo = {}.compare_by_identity)
      return replacement if Rewrite.same?(node, variable)
      return memo[node] if memo.key?(node)

      replace = ->(child) { substitute_for_sampling(child, variable, replacement, memo) }
      memo[node] = rebuild_for_sampling(node, replace)
    end

    def rebuild_for_sampling(node, replace)
      case node
      when Moissanite::BinOp then node.with(lhs: replace.call(node.lhs), rhs: replace.call(node.rhs))
      when Moissanite::Neg, Moissanite::Cast, Moissanite::Not
        node.with(expr: replace.call(node.expr))
      when Moissanite::MathOp then node.with(args: node.args.map(&replace))
      when Moissanite::Select
        node.with(cond: replace.call(node.cond), then_e: replace.call(node.then_e),
                  else_e: replace.call(node.else_e))
      else node
      end
    end

    private_class_method :finite_over_range?, :finite_bounds, :compute_finite_bounds, :arithmetic_bounds,
                         :rounded_bounds, :math_bounds, :simple_math_bounds, :exp_bounds, :contains_select?
  end
end
