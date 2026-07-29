# frozen_string_literal: true

# ∫ を ∂ で検証する — この層の存在理由。
#
#   ruby -Ilib examples/03_integrals_verified_by_differentiation.rb
#
# 表引きの CAS は積分の答えを返して終わり、正しさは実装への信頼に帰着する。
# ここには微分器があるので ∫f = F という主張はそのまま F' = f という
# 検査可能な命題になり、記号的に一致した分は証明書になる。
$LOAD_PATH.unshift(File.expand_path('../lib', __dir__))
sibling = File.expand_path('../../moissanite/lib', __dir__)
$LOAD_PATH.unshift(sibling) if File.directory?(sibling)
require 'nabla'

x = Moissanite::Param.new(type: :f64, name: :x, index: 0)
y = Moissanite::Param.new(type: :f64, name: :y, index: 1)

CASES = {
  '3x^2 + 2x + 1' => (3.0 * x * x) + (2.0 * x) + 1.0,
  '(x+1)(x-2)' => (x + 1.0) * (x - 2.0),
  'sin(2x+1)' => ((2.0 * x) + 1.0).sin,
  'exp(-x)' => (-x).exp,
  '5cos(3x)' => 5.0 * (3.0 * x).cos,
  '1/(2x+3)' => 1.0 / ((2.0 * x) + 3.0),
  'x^3/4 - 7' => ((x * x * x) / 4.0) - 7.0,
  '2' => Moissanite::Expr.lift(2.0, :f64)
}.freeze

puts '∫ f dx = F, then check F\' against f'
puts
symbolic = 0
CASES.each do |label, integrand|
  primitive = Nabla.integrate(integrand, x)
  result = Nabla.verify_integral(integrand, primitive, x, samples: 1000)
  symbolic += 1 if result.symbolic?
  puts format('  ∫ %-16<label>s  %-34<check>s', label: label, check: result.to_s)
  raise "verification failed for #{label}" unless result.ok?
end

puts
puts format('  %<n>d of %<total>d verified symbolically — that is a certificate, not a sample:',
            n: symbolic, total: CASES.size)
puts '  F\' reduced to f syntactically, so the integral is correct as long as the'
puts '  differentiation rules are. The rest agreed numerically, which only means'
puts '  no counterexample was found. Raising this ratio is what improving the'
puts '  simplifier buys.'

# 解けない形は断る。近似を返すのが最悪で、呼び手から見て成功と区別できない。
puts
puts '  refused (no rule — reported rather than approximated):'
[['sin(x)cos(x)', x.sin * x.cos], ['1/(x^2+1)', 1.0 / ((x * x) + 1.0)], ['sin(x^2)', (x * x).sin]].each do |label, f|
  Nabla.integrate(f, x)
  raise "#{label} should have been refused"
rescue Nabla::NotElementary => e
  puts format('    ∫ %-14<label>s %<reason>s', label: label, reason: e.message)
end

# 重積分は入れ子でよい。内側の結果が式として返るため。
puts
puts '  multiple integrals are just nesting, because the inner result is an expression:'
inner = Nabla.definite_integral((x * x) + (y * y), x, 0.0, 1.0)
outer = Nabla.definite_integral(inner, y, 0.0, 1.0)
puts format('    ∫_0^1 ∫_0^1 (x^2 + y^2) dx dy = %<got>.12f   (exact %<want>.12f)',
            got: outer.value, want: 2.0 / 3)
raise 'double integral disagreed' unless (outer.value - (2.0 / 3)).abs < 1e-12

# 極限は微分器の副産物として出てくる。
puts
puts '  limits fall out of the differentiator (L\'Hopital):'
{ 'sin(x)/x' => [x.sin, x, 1.0],
  '(1-cos x)/x^2' => [1.0 - x.cos, x * x, 0.5],
  'sin(3x)/sin(2x)' => [(3.0 * x).sin, (2.0 * x).sin, 1.5] }.each do |label, (num, den, want)|
  got = Nabla.limit(num, den, x, at: 0.0)
  puts format('    lim_{x->0} %-16<label>s = %-6<got>s (exact %<want>s)', label: label, got: got, want: want)
  raise "limit #{label} disagreed" unless (got - want).abs < 1e-9
end
