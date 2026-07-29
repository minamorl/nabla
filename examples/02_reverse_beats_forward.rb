# frozen_string_literal: true

# 逆モードが前進モードに勝つ理由を、規模そのもので見せる。
#
#   ruby -Ilib examples/02_reverse_beats_forward.rb
#
# 前進モードは入力ごとに 1 走査なので全偏微分の合計が O(N^2)。逆モードは
# 1 走査で全部出るので O(N)。そして共有を展開すると逆モードの利得が消える
# ことも同時に示す (Emitter が飾りではなく必須である理由)。
$LOAD_PATH.unshift(File.expand_path('../lib', __dir__))
sibling = File.expand_path('../../moissanite/lib', __dir__)
$LOAD_PATH.unshift(sibling) if File.directory?(sibling)
require 'nabla'

# f(x) = log(eps + sum (x_i - c_i)^2) * (1 + sin(x_0))
def sample(buffer, count)
  squares = (0...count).inject(Moissanite::Expr.lift(1e-6, :f64)) do |acc, i|
    d = buffer[i] - (0.5 + (i * 0.01))
    acc + (d * d)
  end
  squares.log * (1.0 + buffer[0].sin)
end

buffer = Moissanite::Param.new(type: :f64_buf, name: :xs, index: 0)

puts '  inputs   forward nodes   reverse nodes   ratio   reverse as a tree'
[1, 2, 5, 10, 20, 50].each do |count|
  root = sample(buffer, count)
  targets = (0...count).map { |i| buffer[i] }

  forward = targets.sum { |target| Nabla.size(Nabla.derivative(root, target)) }
  partials = Nabla.gradient(root, targets)
  combined = partials.values.reduce(:+)
  reverse = Nabla.size(combined)
  expanded = partials.values.sum { |partial| Nabla.tree_size(partial) }

  puts format('  %-8<n>d %15<f>d %15<r>d %7.2<ratio>fx %19<t>d',
              n: count, f: forward, r: reverse, ratio: forward.to_f / reverse, t: expanded)
end

puts
puts '  forward grows quadratically, reverse linearly — one traversal yields every partial.'
puts '  the last column is what the reverse-mode DAG becomes if its sharing is expanded:'
puts '  emitting without Nabla::Emitter would hand all of that back to the compiler.'

# 規模だけでなく値も確かめる。両モードは随伴の加算順序が違うので最終 ulp が
# ずれうるが、それは浮動小数点の性質であって欠陥ではない。
count = 5
point = Moissanite::Buffer.f64([0.9, 0.3, 1.4, -0.2, 2.7])
reverse_out = Moissanite::Buffer.f64(count)
forward_out = Moissanite::Buffer.f64(count)

Moissanite.kernel(:rev, o: :f64_buf, buf: :f64_buf, n: :i64) do |k, o, buf, _n|
  Nabla.emit_gradient(k, sample(buf, count), (0...count).map { |i| buf[i] })
       .each_with_index { |partial, i| k.store(o, i, partial) }
  k.ret 0
end.call(reverse_out, point, count)

Moissanite.kernel(:fwd, o: :f64_buf, buf: :f64_buf, n: :i64) do |k, o, buf, _n|
  root = sample(buf, count)
  (0...count).each { |i| k.store(o, i, Nabla.derivative(root, buf[i])) }
  k.ret 0
end.call(forward_out, point, count)

worst_ulps = (0...count).map do |i|
  next 0 if reverse_out[i] == forward_out[i]

  ((reverse_out[i] - forward_out[i]).abs / [forward_out[i].abs, 1.0].max / Float::EPSILON).round
end.max

puts
puts "  reverse vs forward, worst disagreement: #{worst_ulps} ulp (association order, not a defect)"
raise 'the two modes disagree by more than rounding' unless worst_ulps <= 4
