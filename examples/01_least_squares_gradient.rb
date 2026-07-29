# frozen_string_literal: true

# 縮約ループを跨ぐ勾配 — テープなし、全パラメータを 1 回の native 呼び出しで。
#
#   ruby -Ilib examples/01_least_squares_gradient.rb
#
# 不変条件を自分で検査するので、正常終了することがそのままテストになる。
$LOAD_PATH.unshift(File.expand_path('../lib', __dir__))
sibling = File.expand_path('../../moissanite/lib', __dir__)
$LOAD_PATH.unshift(sibling) if File.directory?(sibling)
require 'nabla'

DIMENSION = 8
SAMPLES = 200_000

def prediction(weights, features, base)
  (0...DIMENSION).inject(Moissanite::Expr.lift(0.0, :f64)) do |sum, j|
    sum + (weights[j] * features[base + j])
  end
end

# loss = sum_i (w . x_i - y_i)^2
loss = Moissanite.kernel(:loss, w: :f64_buf, x: :f64_buf, y: :f64_buf, n: :i64) do |k, w, x, y, n|
  acc = k.let(0.0)
  k.count(n) do |i|
    base = k.let(i * DIMENSION)
    residual = k.let(prediction(w, x, base) - y[i])
    k.assign(acc, acc + (residual * residual))
  end
  k.ret acc
end

# 和による縮約では各反復の寄与に対する随伴がちょうど 1 なので、
# 「損失の勾配 = 各項の勾配の和」。ループ本体に逆モードをかけて累積する
# だけでよく、ループを記録する必要がない。
gradient = Moissanite.kernel(:loss_grad,
                             g: :f64_buf, w: :f64_buf, x: :f64_buf, y: :f64_buf, n: :i64) do |k, g, w, x, y, n|
  accumulators = (0...DIMENSION).map { k.let(0.0) }
  k.count(n) do |i|
    base = k.let(i * DIMENSION)
    residual = prediction(w, x, base) - y[i] # 素の式で組む。共有は Emitter が戻す。
    Nabla.emit_gradient(k, residual * residual, (0...DIMENSION).map { |j| w[j] })
         .each_with_index { |partial, j| k.assign(accumulators[j], accumulators[j] + partial) }
  end
  (0...DIMENSION).each { |j| k.store(g, j, accumulators[j]) }
  k.ret 0
end

random = Random.new(7)
weights = Moissanite::Buffer.f64(Array.new(DIMENSION) { random.rand - 0.5 })
features = Moissanite::Buffer.f64(Array.new(SAMPLES * DIMENSION) { random.rand - 0.5 })
labels = Moissanite::Buffer.f64(Array.new(SAMPLES) { random.rand - 0.5 })
out = Moissanite::Buffer.f64(DIMENSION)

puts "backend: #{gradient.backend_name}, #{SAMPLES} samples x #{DIMENSION} parameters"

def bench(label, reps = 5)
  yield
  best = (1..reps).map do
    started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    yield
    Process.clock_gettime(Process::CLOCK_MONOTONIC) - started
  end.min
  puts format('  %-42<label>s %7.2<ms>f ms', label: label, ms: best * 1000)
  best
end

value_time = bench('loss only') { loss.call(weights, features, labels, SAMPLES) }
grad_time = bench("gradient, all #{DIMENSION} parameters") { gradient.call(out, weights, features, labels, SAMPLES) }
puts format('  gradient costs %<ratio>.2fx the loss — and that ratio does not grow with the parameter count',
            ratio: grad_time / value_time)

# 解析解: dL/dw_j = sum_i 2 * (w.x_i - y_i) * x_ij
check_samples = 2000
gradient.call(out, weights, features, labels, check_samples)
analytic = Array.new(DIMENSION, 0.0)
check_samples.times do |i|
  base = i * DIMENSION
  residual = (0...DIMENSION).sum { |j| weights[j] * features[base + j] } - labels[i]
  (0...DIMENSION).each { |j| analytic[j] += 2 * residual * features[base + j] }
end
worst = (0...DIMENSION).map { |j| ((out[j] - analytic[j]) / analytic[j]).abs }.max
puts format("\n  max relative error vs the analytic gradient: %<err>.3e", err: worst)
raise 'gradient disagreed with the analytic solution' unless worst < 1e-12

oracle = Moissanite::Buffer.f64(DIMENSION)
gradient.interpret(oracle, weights, features, labels, check_samples)
puts "  oracle == native, bit-for-bit             : #{oracle.to_a == out.to_a}"
raise 'derivative kernel disagreed with the oracle' unless oracle.to_a == out.to_a
